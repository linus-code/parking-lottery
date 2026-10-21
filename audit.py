#!/usr/bin/env python3
"""
对成品的对抗性审查（三组硬检验）
1. 推荐号 029141 在自助重采样下稳定吗？还是噪声里的一个点？
2. 分位序列有自相关吗？有的话卡方和置信区间都被高估了
3. 用「成品模型的构建方式」重做样本外验证（之前验证的是另一个实现）
"""
import json, numpy as np

M, N, BW = 1_000_000, 100_000, 10
SYMS = ["sh600690", "sh688139", "sz300143"]
NAMES = {"sh600690": "海尔智家", "sh688139": "海尔生物", "sz300143": "盈康生命"}
data = json.load(open("data/klines.json"))
rng = np.random.default_rng(20261021)

S = {}
for s in SYMS:
    a = []
    for day, close in data[s]["rows"]:
        c = int(round(close * 100 + 1e-9))
        a.append((day, (c // 10) % 10, c % 10))
    S[s] = a

def marg(days_idx, lam=120.0):
    """给定某只股票的样本索引 -> 因子化 100 格表（pi = fen*10+jiao）"""
    pm = []
    for s in SYMS:
        a = np.array([(S[s][i][1], S[s][i][2]) for i in days_idx[s]])
        n = len(a)
        pf = (np.bincount(a[:, 1], minlength=10) + lam * 0.1) / (n + lam)
        pj = (np.bincount(a[:, 0], minlength=10) + lam * 0.1) / (n + lam)
        pm.append(np.multiply.outer(pf, pj).ravel())
    return pm

def build_P(pm):
    P = np.zeros(N)
    g3 = np.zeros(10)
    for p in range(100):
        g3[p // 10] += pm[2][p]
    for a in range(100):
        for b in range(100):
            base = a * 1000 + b * 10
            P[base:base + 10] += pm[0][a] * pm[1][b] * g3
    return P / P.sum()

def ring_E(P):
    M3, OFF = 3 * N, N // 2
    Pe = np.tile(P, 3)
    SA = np.concatenate([[0.0], np.cumsum(Pe)])
    SB = np.concatenate([[0.0], np.cumsum(Pe * np.arange(M3))])
    n = np.arange(N); p = n + N
    r = (SB[p + OFF + 1] - SB[p + 1]) - p * (SA[p + OFF + 1] - SA[p + 1])
    l = p * (SA[p] - SA[p - OFF]) - (SB[p] - SB[p - OFF])
    return (r + l) * BW

all_idx = {s: list(range(len(S[s]))) for s in SYMS}
P_full = build_P(marg(all_idx))
E_full = ring_E(P_full)
best = int(np.argmin(E_full))
print("=" * 76)
print("基准：全样本最优号 = %06d   E=%.0f" % (best * BW + (best * 7 + 3) % 10, E_full[best]))
print("=" * 76)

# ---------- 1. 块自助重采样 ----------
print("\n【检验一】块自助重采样 300 次：最优号到底有多稳？")
BLOCK = 20
picks = []
for r in range(300):
    idx = {}
    for s in SYMS:
        n = len(S[s]); k = int(np.ceil(n / BLOCK))
        starts = rng.integers(0, max(1, n - BLOCK), size=k)
        seq = np.concatenate([np.arange(st, min(st + BLOCK, n)) for st in starts])[:n]
        idx[s] = seq
    P = build_P(marg(idx))
    picks.append(int(np.argmin(ring_E(P))))
picks = np.array(picks)
nums = picks * BW + (picks * 7 + 3) % 10
print(f"  最优号落在 {nums.min():06d} ~ {nums.max():06d}，跨度 {nums.max()-nums.min():,}")
print(f"  中位数 {int(np.median(nums)):06d}，四分位 {int(np.percentile(nums,25)):06d} / {int(np.percentile(nums,75)):06d}")
print(f"  在 0–10 万带内的比例：{(picks < 10000).mean()*100:.1f}%")
for tol in (5000, 10000, 20000):
    frac = (np.abs(nums - (best * BW)) <= tol).mean()
    print(f"  落在基准号 ±{tol:,} 内的比例：{frac*100:5.1f}%")

boot = {
    "n": int(len(nums)),
    "min": int(nums.min()), "max": int(nums.max()),
    "median": int(np.median(nums)),
    "q25": int(np.percentile(nums, 25)), "q75": int(np.percentile(nums, 75)),
    "in_band_pct": round(float((picks < 10000).mean() * 100), 1),
    "hit_5k_pct": round(float((np.abs(nums - best * BW) <= 5000).mean() * 100), 1),
    "hit_10k_pct": round(float((np.abs(nums - best * BW) <= 10000).mean() * 100), 1),
    "hit_20k_pct": round(float((np.abs(nums - best * BW) <= 20000).mean() * 100), 1),
}
json.dump(boot, open("data/audit.json", "w"), ensure_ascii=False, indent=1)
print("  → 已写入 data/audit.json")

# ---------- 2. 自相关 / 有效样本量 ----------
print("\n【检验二】分位=0 的指示序列有自相关吗？（有则卡方被高估）")
for s in SYMS:
    x = np.array([1.0 if f == 0 else 0.0 for _, j, f in S[s]])
    n = len(x); mu = x.mean()
    acf = []
    for lag in range(1, 6):
        c = np.corrcoef(x[:-lag], x[lag:])[0, 1]
        acf.append(c)
    rho1 = acf[0]
    neff = n * (1 - rho1) / (1 + rho1) if rho1 > -1 else n
    se_iid = np.sqrt(mu * (1 - mu) / n)
    se_eff = np.sqrt(mu * (1 - mu) / max(neff, 1))
    print(f"  {NAMES[s]:6s} n={n:4d}  p0={mu*100:5.2f}%  ACF(1..5)=" +
          " ".join(f"{c:+.3f}" for c in acf))
    print(f"          ρ1={rho1:+.3f} → 有效样本 n_eff≈{neff:,.0f}   "
          f"SE: iid {se_iid*100:.2f}% → 修正 {se_eff*100:.2f}%  "
          f"(膨胀 {se_eff/se_iid:.2f}×)")

# ---------- 3. 用成品构建方式重做样本外 ----------
print("\n【检验三】按成品方式重做样本外（每只股票用自己的全部历史，而非仅共同交易日）")
common = None
for s in SYMS:
    d = set(x[0] for x in S[s])
    common = d if common is None else (common & d)
common = sorted(common)

def idx_before(s, cut):
    return [i for i, x in enumerate(S[s]) if x[0] < cut]

for cut in ["2023-01-02", "2024-01-02", "2025-01-02"]:
    tr_idx = {s: idx_before(s, cut) for s in SYMS}
    P = build_P(marg(tr_idx))
    E = ring_E(P)
    b = int(np.argmin(E))
    te = [d for d in common if d >= cut]
    dmap = {s: {x[0]: (x[1], x[2]) for x in S[s]} for s in SYMS}
    X = np.array([(lambda p: p[0] * 10000 + p[1] * 100 + p[2])(
        [dmap[s][d][1] * 10 + dmap[s][d][0] for s in SYMS]) for d in te])
    dd = np.minimum(np.abs(b * BW - X), M - np.abs(b * BW - X))
    row = []
    for t in (50_000, 83_333, 150_000):
        p = float((dd < t).mean())
        row.append(f"t={t//1000}k: {p*100:5.2f}% ({p/(t/M):.2f}×)")
    print(f"  切点 {cut}  训练各股 {[len(tr_idx[s]) for s in SYMS]}  测试 {len(te)} 天  "
          f"号 {b*BW+(b*7+3)%10:06d}")
    print(f"     " + "   ".join(row))
