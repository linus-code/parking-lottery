#!/usr/bin/env python3
"""补充验证：(1) 固定阈值下的样本外 CDF（低方差）  (2) 校验 O(M) 前缀和公式，准备移植到前端"""
import json, numpy as np

M, OFF = 1_000_000, 500_000
SYMS = ["sh600690", "sh688139", "sz300143"]
data = json.load(open("data/klines.json"))

def series(sym):
    out = []
    for day, close in data[sym]["rows"]:
        c = int(round(close * 100 + 1e-9))
        out.append((day, (c // 10) % 10, c % 10))
    return out
S = {s: series(s) for s in SYMS}
common = None
for s in SYMS:
    d = set(x[0] for x in S[s]); common = d if common is None else (common & d)
common = sorted(common)

def P_from(days, lam=120.0):
    pm = []
    for s in SYMS:
        dmap = {x[0]: (x[1], x[2]) for x in S[s]}
        a = np.array([dmap[d] for d in days]); n = len(a)
        pf = (np.bincount(a[:, 1], minlength=10) + lam * 0.1) / (n + lam)
        pj = (np.bincount(a[:, 0], minlength=10) + lam * 0.1) / (n + lam)
        pm.append(np.multiply.outer(pf, pj).ravel())
    return pm

def build_P(pm):
    return np.multiply.outer(np.multiply.outer(pm[0], pm[1]).ravel(), pm[2]).ravel()

def X_of(days):
    dmap = {s: {x[0]: (x[1], x[2]) for x in S[s]} for s in SYMS}
    return np.array([(lambda p: p[0]*10000 + p[1]*100 + p[2])(
        [dmap[s][d][1]*10 + dmap[s][d][0] for s in SYMS]) for d in days])

# ---------- (1) 固定阈值样本外 CDF ----------
print("=" * 76)
print("一、固定阈值下的样本外命中率（低方差，直接看机制是否成立）")
print("=" * 76)
for cut in ["2023-01-02", "2024-01-02", "2025-01-02"]:
    tr = [d for d in common if d < cut]; te = [d for d in common if d >= cut]
    Ptr = build_P(P_from(tr))
    k = np.minimum(np.arange(M), M - np.arange(M)).astype(float)
    E = np.fft.irfft(np.fft.rfft(Ptr) * np.fft.rfft(k), n=M)
    best = int(np.argmin(E))
    Xte = X_of(te)
    d = np.minimum(np.abs(best - Xte), M - np.abs(best - Xte))
    print(f"\n切点 {cut}  训练{len(tr)}天 测试{len(te)}天  号码 {best:06d}")
    for t in (50_000, 83_333, 150_000):
        p = float((d < t).mean())
        se = float(np.sqrt(p * (1 - p) / len(te)))
        print(f"   t={t:>7,}  样本外命中 {p*100:5.2f}% ±{se*100:.1f}  "
              f"均匀基准 {t/M*100:5.2f}%   倍数 {p/(t/M):4.2f}x")

# ---------- (2) 校验 O(M) 前缀和公式 ----------
print("\n" + "=" * 76)
print("二、校验 O(M) 前缀和算法（前端要用，必须与 FFT 结果一致）")
print("=" * 76)

def ring_E_fft(P):
    k = np.minimum(np.arange(M), M - np.arange(M)).astype(float)
    return np.fft.irfft(np.fft.rfft(P) * np.fft.rfft(k), n=M)

def ring_E_prefix(P):
    """E(n) = Σ_j P(j)·d(n,j)，用 3M 前缀和 O(M) 求解"""
    Pext = np.tile(P, 3)
    SA = np.concatenate([[0.0], np.cumsum(Pext)])
    SB = np.concatenate([[0.0], np.cumsum(Pext * np.arange(3 * M))])
    n = np.arange(M)
    p = n + M
    # 右半环 j∈(n, n+OFF]，贡献 (j-n)；Pext 下标 i=j+M，故 j-n = i-p
    right = (SB[p + OFF + 1] - SB[p + 1]) - p * (SA[p + OFF + 1] - SA[p + 1])
    # 左半环 j∈[n-OFF, n)，贡献 (n-j)；i=j+M，故 n-j = p-i
    left = p * (SA[p] - SA[p - OFF]) - (SB[p] - SB[p - OFF])
    return right + left

pm = P_from(common)
P = build_P(pm)
E1 = ring_E_fft(P)
E2 = ring_E_prefix(P)
print(f"FFT 最优号 {int(np.argmin(E1)):06d}  E={E1.min():,.2f}")
print(f"前缀 最优号 {int(np.argmin(E2)):06d}  E={E2.min():,.2f}")
print(f"两法最大绝对偏差 {np.abs(E1-E2).max():.6e}")
print(f"两法最大相对偏差 {(np.abs(E1-E2)/np.maximum(E1,1)).max():.3e}")

# 降采样版本（N=100000，bin 宽 10）——前端实际使用
N = 100_000
Pb = P.reshape(N, 10).sum(1)
Nb, OFFb = N, N // 2
def ring_E_prefix_n(P, N):
    OFFn = N // 2
    Pext = np.tile(P, 3)
    SA = np.concatenate([[0.0], np.cumsum(Pext)])
    SB = np.concatenate([[0.0], np.cumsum(Pext * np.arange(3 * N))])
    n = np.arange(N); p = n + N
    return ((SB[p+OFFn+1] - SB[p+1]) - p*(SA[p+OFFn+1] - SA[p+1])) + \
           (p*(SA[p] - SA[p-OFFn]) - (SB[p] - SB[p-OFFn]))
Eb = ring_E_prefix_n(Pb, Nb)
print(f"\n降采样 N={Nb} (bin=10) 最优 bin={int(np.argmin(Eb))}  → 号码约 {int(np.argmin(Eb))*10:06d}")
print(f"全分辨率最优号 {int(np.argmin(E1)):06d}，两者相差 "
      f"{abs(int(np.argmin(Eb))*10 - int(np.argmin(E1))):,} 步")
