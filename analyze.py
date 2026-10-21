#!/usr/bin/env python3
"""
车位摇号 · 核心建模
规则：X = 三只股票收盘价末两位反转后拼成的 6 位数；中签 = 环形距离 d(n,X) 排名前 N

关键：中签是"排名"问题，正确目标不是最小化 E[d]，而是最大化 P(d(n,X) < 阈值)。
阈值 ≈ 500000 * 车位/人数，等价于"把号码放在 X 分布最密的弧段里"。
"""
import json, numpy as np

M, HALF = 1_000_000, 500_000
SYMS = ["sh600690", "sh688139", "sz300143"]
NAMES = {"sh600690": "海尔智家", "sh688139": "海尔生物", "sz300143": "盈康生命"}
CUR = {"sh600690": 19.69, "sh688139": 32.69, "sz300143": 7.73}

data = json.load(open("data/klines.json"))
rng = np.random.default_rng(20261021)

# ---------- 1. 末两位分布 ----------
dig = {}
for sym in SYMS:
    d = []
    for day, close in data[sym]["rows"]:
        c = int(round(close * 100 + 1e-9))
        d.append(((c // 10) % 10, c % 10))        # (角位, 分位)
    dig[sym] = np.array(d)

print("=" * 80)
print("一、末两位分布检验")
print("=" * 80)
tabs = {}
for sym in SYMS:
    a = dig[sym]; n = len(a)
    jiao, fen = a[:, 0], a[:, 1]
    cf = np.bincount(fen, minlength=10) / n
    cj = np.bincount(jiao, minlength=10) / n
    # 10x10 独立性检验
    obs = np.zeros((10, 10))
    for j, f in a:
        obs[j, f] += 1
    row, col = obs.sum(1, keepdims=True), obs.sum(0, keepdims=True)
    exp = row @ col / n
    chi_ind = float(((obs - exp) ** 2 / exp).sum())
    chi_fen = float((((cf * n - n / 10) ** 2) / (n / 10)).sum())
    chi_jiao = float((((cj * n - n / 10) ** 2) / (n / 10)).sum())
    tabs[sym] = {"fen": cf, "jiao": cj, "n": n}
    print(f"\n{NAMES[sym]}  n={n}")
    print("  分位 " + " ".join(f"{v*100:5.1f}" for v in cf) + "   χ²="
          f"{chi_fen:6.1f} {'显著' if chi_fen>21.67 else '不显著'}")
    print("  角位 " + " ".join(f"{v*100:5.1f}" for v in cj) + "   χ²="
          f"{chi_jiao:6.1f} {'显著' if chi_jiao>21.67 else '不显著'}")
    print(f"  角位×分位 独立性 χ²={chi_ind:6.1f} (df=81, 95%临界103.0) "
          f"{'→ 相关，用全表' if chi_ind>103 else '→ 独立，可用因子化'}")

# ---------- 2. 给定当前价的末两位预测（粘性效应）----------
print("\n" + "=" * 80)
print("二、给定当前收盘价，预测开奖日末两位（粘性效应 + 聚集偏差）")
print("=" * 80)

def bias_table(sym, lam=120.0):
    """聚集偏差：经验频率向均匀收缩，并因子化到 100 格，索引 pi = 分位*10+角位"""
    t = tabs[sym]; n = t["n"]
    pf = (t["fen"] * n + lam * 0.1) / (n + lam)
    pj = (t["jiao"] * n + lam * 0.1) / (n + lam)
    return np.multiply.outer(pf, pj).ravel()          # [fen, jiao] -> pi

def cond_table(sym, cur, nsamp=300_000):
    """2 个交易日后收盘价的末两位分布（假设对数正态、零漂移）"""
    rows = data[sym]["rows"]
    px = np.array([r[1] for r in rows[-260:]])
    ret = np.diff(np.log(px))
    sd2 = float(np.std(ret, ddof=1)) * np.sqrt(2.0)
    close2 = cur * np.exp(rng.normal(0, sd2, nsamp))
    cents = np.round(close2 * 100).astype(np.int64)
    pi = (cents % 10) * 10 + ((cents // 10) % 10)
    cnt = np.bincount(pi, minlength=100).astype(float)
    return cnt / cnt.sum(), sd2

P_stocks, info = [], []
for sym in SYMS:
    cond, sd2 = cond_table(sym, CUR[sym])
    b = bias_table(sym)
    p = cond * b
    p /= p.sum()
    P_stocks.append(p)
    # 单独看两种效应各自的影响
    print(f"{NAMES[sym]}: 2日波动率 σ={sd2*100:.2f}%  "
          f"分位众数(纯粘性)={int(cond.reshape(10,10).sum(1).argmax())}  "
          f"分位众数(合并)={int(p.reshape(10,10).sum(1).argmax())}")
    info.append({"sym": sym, "sd2": sd2, "cond_fen": cond.reshape(10,10).sum(1).tolist()})

# ---------- 3. 环形分析 ----------
def build_P(pm):
    P = np.multiply.outer(pm[0], pm[1]).ravel()
    return np.multiply.outer(P, pm[2]).ravel()

P = build_P(P_stocks)
assert abs(P.sum() - 1) < 1e-9

def ring_E(P):
    k = np.minimum(np.arange(M), M - np.arange(M)).astype(float)
    return np.fft.irfft(np.fft.rfft(P) * np.fft.rfft(k), n=M)

def window_mass(P, half):
    """每个 n 落在 [n-half, n+half] 环形窗口内的概率"""
    P2 = np.concatenate([P, P])
    C = np.concatenate([[0.0], np.cumsum(P2)])
    n = np.arange(M)
    return C[n + 2 * half + 1] - C[n]

E = ring_E(P)
print("\n" + "=" * 80)
print("三、两种目标函数给出的最优号")
print("=" * 80)
b_E = int(np.argmin(E))
print(f"最小化 E[环形距离]:  {b_E:06d}   E[d]={E[b_E]:,.0f}  (均匀基准 250,000，改善 "
      f"{(250000-E[b_E])/250000*100:.2f}%)")

print("\n最大化落入阈值窗口的概率（这才是决定中签的）：")
results = {}
for players, spots in [(300, 20), (300, 50), (400, 50), (400, 100)]:
    half = int(round(HALF * spots / players))
    wm = window_mass(P, half)
    bn = int(np.argmax(wm))
    results[(players, spots)] = (bn, float(wm[bn]), half)
    print(f"  {players}人/{spots}位  阈值半宽 ±{half:,}   最优号 {bn:06d}   "
          f"命中窗口概率 {wm[bn]*100:.2f}%  (随机号 {2*half/M*100:.2f}%)")

# ---------- 4. 蒙特卡洛验证 ----------
print("\n" + "=" * 80)
print("四、蒙特卡洛验证：选号策略的真实中签率")
print("=" * 80)
NS = 400_000
Xs = rng.choice(M, size=NS, p=P / P.sum())

def win_rate(pick, players, spots, trials=8000):
    X = rng.choice(M, size=trials, p=P / P.sum())
    comp = rng.integers(0, M, size=(trials, players - 1))
    dc = np.minimum(np.abs(comp - X[:, None]), M - np.abs(comp - X[:, None]))
    thr = np.partition(dc, spots - 1, axis=1)[:, spots - 1]
    dme = np.minimum(np.abs(pick - X), M - np.abs(pick - X))
    return float((dme < thr).mean())

for (players, spots), (bn, wp, half) in results.items():
    r_best = win_rate(bn, players, spots)
    r_E = win_rate(b_E, players, spots)
    r_rand = win_rate(int(rng.integers(0, M)), players, spots)
    print(f"  {players}人/{spots}位:  窗口最优号 {r_best*100:5.2f}%   "
          f"最小E号 {r_E*100:5.2f}%   随机号 {r_rand*100:5.2f}%   "
          f"（理论基准 {spots/players*100:.2f}%）")

# ---------- 5. 保存 ----------
model = {
    "generated": "2026-10-21",
    "stocks": [{"sym": s, "name": NAMES[s], "n": tabs[s]["n"],
                "fen": [round(float(x), 6) for x in tabs[s]["fen"]],
                "jiao": [round(float(x), 6) for x in tabs[s]["jiao"]],
                "sd2": info[i]["sd2"]} for i, s in enumerate(SYMS)],
    "best_ring_E": b_E, "E_best": float(E[b_E]),
    "results": [{"players": k[0], "spots": k[1], "best": v[0],
                 "winprob": v[1], "half": v[2]} for k, v in results.items()],
}
json.dump(model, open("data/model.json", "w"), ensure_ascii=False, indent=1)
print("\nsaved data/model.json")
