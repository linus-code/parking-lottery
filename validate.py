#!/usr/bin/env python3
"""对抗性验证：这个"优势"是真信号还是 100 万选 1 的过拟合？"""
import json, numpy as np

M, HALF = 1_000_000, 500_000
SYMS = ["sh600690", "sh688139", "sz300143"]
NAMES = {"sh600690": "海尔智家", "sh688139": "海尔生物", "sz300143": "盈康生命"}
data = json.load(open("data/klines.json"))
rng = np.random.default_rng(7)

def series(sym):
    rows = data[sym]["rows"]
    out = []
    for day, close in rows:
        c = int(round(close * 100 + 1e-9))
        out.append((day, (c // 10) % 10, c % 10))
    return out

S = {s: series(s) for s in SYMS}

# ---------- 1. 分位分布的稳定性：前后两半是否一致 ----------
print("=" * 82)
print("一、稳定性检验：把历史劈成两半，分位分布还一样吗？")
print("=" * 82)
for s in SYMS:
    a = np.array([(j, f) for _, j, f in S[s]])
    h = len(a) // 2
    f1 = np.bincount(a[:h, 1], minlength=10) / h
    f2 = np.bincount(a[h:, 1], minlength=10) / (len(a) - h)
    corr = float(np.corrcoef(f1, f2)[0, 1])
    # 同质性卡方
    n1, n2 = h, len(a) - h
    o = np.vstack([f1 * n1, f2 * n2])
    e = o.sum(1, keepdims=True) @ o.sum(0, keepdims=True) / o.sum()
    chi = float(((o - e) ** 2 / e).sum())
    print(f"{NAMES[s]:6s} 前半(n={n1:4d}) 分位0占比 {f1[0]*100:5.1f}%  "
          f"后半(n={n2:4d}) {f2[0]*100:5.1f}%   两半相关 r={corr:+.3f}   "
          f"同质性χ²={chi:5.1f} {'→ 一致' if chi < 16.92 else '→ 漂移'}")

# ---------- 2. 共同交易日 ----------
common = None
for s in SYMS:
    d = set(x[0] for x in S[s])
    common = d if common is None else (common & d)
common = sorted(common)
print(f"\n三只股票共同交易日: {len(common)} 天  {common[0]} ~ {common[-1]}")

# ---------- 3. 训练/测试切分 ----------
def P_from(days, lam=120.0):
    pm = []
    for s in SYMS:
        dmap = {x[0]: (x[1], x[2]) for x in S[s]}
        a = np.array([dmap[d] for d in days])
        n = len(a)
        pf = (np.bincount(a[:, 1], minlength=10) + lam * 0.1) / (n + lam)
        pj = (np.bincount(a[:, 0], minlength=10) + lam * 0.1) / (n + lam)
        pm.append(np.multiply.outer(pf, pj).ravel())
    return pm

def build_P(pm):
    return np.multiply.outer(np.multiply.outer(pm[0], pm[1]).ravel(), pm[2]).ravel()

def ring_E(P):
    k = np.minimum(np.arange(M), M - np.arange(M)).astype(float)
    return np.fft.irfft(np.fft.rfft(P) * np.fft.rfft(k), n=M)

def X_of(days):
    dmap = {s: {x[0]: (x[1], x[2]) for x in S[s]} for s in SYMS}
    out = []
    for d in days:
        pi = [(dmap[s][d][1] * 10 + dmap[s][d][0]) for s in SYMS]
        out.append(pi[0] * 10000 + pi[1] * 100 + pi[2])
    return np.array(out)

def win_rate(pick, X, players, spots, reps=400):
    if pick is None:
        return None
    dme = np.minimum(np.abs(pick - X), M - np.abs(pick - X))
    hit = 0
    for _ in range(reps):
        comp = rng.integers(0, M, size=(len(X), players - 1))
        dc = np.minimum(np.abs(comp - X[:, None]), M - np.abs(comp - X[:, None]))
        thr = np.partition(dc, spots - 1, axis=1)[:, spots - 1]
        hit += (dme < thr).mean()
    return hit / reps

print("\n" + "=" * 82)
print("二、样本外检验：用前段历史选号，在后段历史里看真实表现")
print("=" * 82)

cuts = ["2023-01-02", "2024-01-02", "2025-01-02"]
for cut in cuts:
    tr = [d for d in common if d < cut]
    te = [d for d in common if d >= cut]
    if len(tr) < 300 or len(te) < 150:
        continue
    P_tr = build_P(P_from(tr))
    E_tr = ring_E(P_tr)
    best = int(np.argmin(E_tr))
    Xte = X_of(te)

    print(f"\n切点 {cut}   训练 {len(tr)} 天 / 测试 {len(te)} 天   训练选出的号 = {best:06d}")
    for players, spots in [(300, 50), (400, 50)]:
        w_best = win_rate(best, Xte, players, spots)
        # 随机号基准：抽 30 个随机号取平均
        wr = [win_rate(int(rng.integers(0, M)), Xte, players, spots, reps=120) for _ in range(30)]
        base = float(np.mean(wr))
        print(f"   {players}人/{spots}位   最优号 {w_best*100:5.2f}%   "
              f"随机号均值 {base*100:5.2f}%   样本外提升 {(w_best-base)*100:+.2f}pp   "
              f"(理论基准 {spots/players*100:.2f}%)")

# ---------- 4. 机制：最优号的距离分布 vs 均匀 ----------
print("\n" + "=" * 82)
print("三、机制检验：最优号为什么占优？看距离的累积分布")
print("=" * 82)
P_all = build_P(P_from(common))
E_all = ring_E(P_all)
best_all = int(np.argmin(E_all))
Xall = X_of(common)
d_best = np.minimum(np.abs(best_all - Xall), M - np.abs(best_all - Xall))
print(f"全样本最优号 {best_all:06d}，E[d]={E_all[best_all]:,.0f}")
print("  阈值 t      最优号命中率    均匀基准     倍数")
for t in (20_000, 50_000, 83_333, 150_000, 250_000):
    p = float((d_best < t).mean())
    print(f"  {t:>7,}      {p*100:6.2f}%        {t/M*100:6.2f}%     {p/(t/M):5.2f}x")
