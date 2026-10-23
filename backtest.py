#!/usr/bin/env python3
"""
走查回测（walk-forward）：完全复刻实际流程
  周三/周四选号 → 模型只用"周三之前"的数据（严格无未来信息）→ 周五收盘开奖

对比四种策略：
  A 推荐号（最小化期望环形距离）
  B 随机号
  C 追昨日号（用周四收盘的派生号）
  D 环中点 500000（零信息基准）
"""
import json, numpy as np
from datetime import datetime

M, N, BW = 1_000_000, 100_000, 10
SYMS = ["sh600690", "sh688139", "sz300143"]
NAMES = {"sh600690": "海尔智家", "sh688139": "海尔生物", "sz300143": "盈康生命"}
data = json.load(open("data/klines.json"))

# 每只股票: 日期 -> (角位, 分位)
S, DATES = {}, {}
for s in SYMS:
    m = {}
    for day, close in data[s]["rows"]:
        ci = int(round(close * 100 + 1e-9))
        m[day] = ((ci // 10) % 10, ci % 10)
    S[s] = m
    DATES[s] = [d for d, _ in data[s]["rows"]]

common = None
for s in SYMS:
    common = set(S[s]) if common is None else (common & set(S[s]))
common = sorted(common)

def p1_of(jiao, fen): return fen * 10 + jiao
def X_of(day):
    return (p1_of(*S[SYMS[0]][day]) * 10000 +
            p1_of(*S[SYMS[1]][day]) * 100 +
            p1_of(*S[SYMS[2]][day]))

def marg_before(cut, lam=120.0):
    """只用 cut 之前的样本（每只股票各用自己的全部历史）"""
    pm = []
    for s in SYMS:
        idx = [i for i, d in enumerate(DATES[s]) if d < cut]
        if len(idx) < 250: return None
        a = np.array([S[s][DATES[s][i]] for i in idx])
        n = len(a)
        pf = (np.bincount(a[:, 1], minlength=10) + lam * 0.1) / (n + lam)
        pj = (np.bincount(a[:, 0], minlength=10) + lam * 0.1) / (n + lam)
        pm.append(np.multiply.outer(pf, pj).ravel())
    return pm

def build_P(pm):
    P = np.zeros(N); g3 = np.zeros(10)
    for p in range(100): g3[p // 10] += pm[2][p]
    for a in range(100):
        pa = pm[0][a]
        for b in range(100):
            base = a * 1000 + b * 10
            P[base:base + 10] += pa * pm[1][b] * g3
    return P / P.sum()

def ringE(P):
    M3, OFF = 3 * N, N // 2
    Pe = np.tile(P, 3)
    SA = np.concatenate([[0.], np.cumsum(Pe)])
    SB = np.concatenate([[0.], np.cumsum(Pe * np.arange(M3))])
    n = np.arange(N); p = n + N
    return ((SB[p + OFF + 1] - SB[p + 1]) - p * (SA[p + OFF + 1] - SA[p + 1])
            + p * (SA[p] - SA[p - OFF]) - (SB[p] - SB[p - OFF])) * BW

def rdist(a, b):
    d = abs(a - b); return min(d, M - d)

# ---------- 找出所有 周三/周四/周五 连续交易日 ----------
weeks = []
for i in range(2, len(common)):
    fri = common[i]
    if datetime.strptime(fri, "%Y-%m-%d").weekday() != 4:
        continue
    wed, thu = common[i - 2], common[i - 1]
    weeks.append((wed, thu, fri))
print(f"可用周数（周三/四/五均为交易日）: {len(weeks)}   区间 {weeks[0][0]} ~ {weeks[-1][2]}")

# ---------- 逐周回测 ----------
rng = np.random.default_rng(20261024)
rows = []
for k, (wed, thu, fri) in enumerate(weeks):
    pm = marg_before(wed)          # 严格：只用周三之前
    if pm is None: continue
    E = ringE(build_P(pm))
    b = int(np.argmin(E))
    rec = b * BW + (b * 7 + 3) % 10

    X = X_of(fri)
    thuX = X_of(thu)
    picks = {
        "A 推荐号": rec,
        "C 追昨日号": thuX,
        "D 环中点": 500000,
    }

    # 中签模拟：299 个随机对手，取第 50 小距离为阈值
    comp = rng.integers(0, M, size=(400, 299))
    dc = np.minimum(np.abs(comp - X), M - np.abs(comp - X))
    thr = np.partition(dc, 49, axis=1)[:, 49]

    d, win = {}, {}
    for k2, v in picks.items():
        d[k2] = rdist(v, X)
        win[k2] = float((d[k2] < thr).mean())

    # 随机号：抽 400 个，距离取均值、中签率取均值（必须是单号对单阈值）
    rp = rng.integers(0, M, size=400)
    rd = np.minimum(np.abs(rp[:, None] - X), M - np.abs(rp[:, None] - X))
    d["B 随机号"] = float(rd.mean())
    win["B 随机号"] = float((rd < thr[:, None]).mean())

    rows.append({"wed": wed, "fri": fri, "X": X, "rec": rec,
                 **{f"d_{k2}": v for k2, v in d.items()},
                 **{f"w_{k2}": v for k2, v in win.items()}})
    if (k + 1) % 60 == 0:
        print(f"  …已完成 {k+1}/{len(weeks)}")

import statistics as st
print(f"\n有效回测周数: {len(rows)}")

print("\n" + "=" * 72)
print("策略对比（300 人 / 50 车位，对手随机选号）")
print("=" * 72)
print(f"{'策略':<12}{'平均距离':>12}{'中位距离':>12}{'中签率':>10}{'相对基准':>10}")
base = np.mean([r["w_B 随机号"] for r in rows])
for k2 in ["A 推荐号", "B 随机号", "C 追昨日号", "D 环中点"]:
    dd = [r[f"d_{k2}"] for r in rows]
    ww = [r[f"w_{k2}"] for r in rows]
    lift = (np.mean(ww) / base - 1) * 100
    print(f"{k2:<12}{np.mean(dd):>12,.0f}{np.median(dd):>12,.0f}"
          f"{np.mean(ww)*100:>9.2f}%{lift:>9.1f}%")

# 推荐号 vs 随机号：逐周胜负
better = sum(1 for r in rows if r["d_A 推荐号"] < r["d_B 随机号"])
print(f"\n推荐号距离优于随机号的周数: {better}/{len(rows)} = {better/len(rows)*100:.1f}%")

# 分年度
print("\n分年度中签率（推荐号 vs 随机号）:")
years = sorted(set(r["fri"][:4] for r in rows))
print(f"{'年份':<8}{'周数':>6}{'推荐号':>10}{'随机号':>10}{'差':>9}")
for y in years:
    sub = [r for r in rows if r["fri"][:4] == y]
    if len(sub) < 10: continue
    a = np.mean([r["w_A 推荐号"] for r in sub]) * 100
    bb = np.mean([r["w_B 随机号"] for r in sub]) * 100
    print(f"{y:<8}{len(sub):>6}{a:>9.2f}%{bb:>9.2f}%{a-bb:>+8.2f}pp")

json.dump(rows, open("data/backtest.json", "w"), ensure_ascii=False)
print("\nsaved data/backtest.json")
