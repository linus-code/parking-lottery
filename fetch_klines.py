#!/usr/bin/env python3
"""拉取三只股票历史日线收盘价（不复权 —— 摇号规则读的就是原始收盘价）"""
import json, time, subprocess, os

STOCKS = [("sh600690", "海尔智家"), ("sh688139", "海尔生物"), ("sz300143", "盈康生命")]
YEARS = list(range(2014, 2027))

def curl(url):
    return subprocess.run(["curl", "-s", "--max-time", "40", "-A", "Mozilla/5.0", url],
                          capture_output=True, text=True).stdout

def fetch(sym):
    rows = {}
    for y in YEARS:
        url = (f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?"
               f"param={sym},day,{y}-01-01,{y}-12-31,640,")
        try:
            d = json.loads(curl(url))
            node = d["data"][sym]
            arr = node.get("day") or node.get("qfqday") or []
            for it in arr:
                rows[it[0]] = float(it[2])   # [date, open, close, high, low, vol]
        except Exception as e:
            print(f"  ! {sym} {y} {type(e).__name__}")
        time.sleep(0.25)
    return sorted(rows.items())

result = {}
for sym, name in STOCKS:
    rs = fetch(sym)
    # 防御性守卫：若数据源日后开始返回当日未收盘 K 线，盘中重训会把"当前价"
    # 当成收盘价喂进模型，污染末两位分布。
    # 实测腾讯 web.ifzq.gtimg.cn 目前不返回当日未收盘 K 线（2026-10-22 盘中验证），
    # 所以这段当前不会触发，仅作数据源变更时的兜底。
    today = time.strftime("%Y-%m-%d")
    now_hm = time.strftime("%H:%M")
    if rs and rs[-1][0] == today and now_hm < "15:05":
        dropped = rs.pop()
        print(f"  ⚠ {name}: 剔除今日未收盘 K 线 {dropped[0]} 收盘={dropped[1]}（当前 {now_hm}，15:05 后重跑可纳入）")
    result[sym] = {"name": name, "rows": rs}
    print(f"{sym} {name}: {len(rs)} 条  {rs[0][0]} ~ {rs[-1][0]}  最新收盘 {rs[-1][1]}")

os.makedirs("data", exist_ok=True)
json.dump(result, open("data/klines.json", "w"), ensure_ascii=False)
print("saved")
