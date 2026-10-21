#!/usr/bin/env python3
"""把 data/model.json + data/audit.json 注入 index.template.html，生成 index.html。

改动模板或模型后必须重跑，否则 index.html 会与模板悄悄分叉。
"""
import json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
tpl_path = os.path.join(ROOT, "index.template.html")
out_path = os.path.join(ROOT, "index.html")

tpl = open(tpl_path, encoding="utf-8").read()
if "__MODEL_JSON__" not in tpl:
    sys.exit("模板里找不到 __MODEL_JSON__ 占位符")

model = json.load(open(os.path.join(ROOT, "data/model.json"), encoding="utf-8"))
audit_path = os.path.join(ROOT, "data/audit.json")
audit = json.load(open(audit_path, encoding="utf-8")) if os.path.exists(audit_path) else None

slim = {
    "generated": model["generated"],
    "rule": model["rule"],
    "stocks": [
        {k: s[k] for k in ("sym", "code", "name", "n", "fen", "jiao", "sd2", "last_close")}
        for s in model["stocks"]
    ],
}
if audit:
    slim["audit"] = audit

js = json.dumps(slim, ensure_ascii=False, separators=(",", ":"))
out = tpl.replace("__MODEL_JSON__", js)
open(out_path, "w", encoding="utf-8").write(out)
print(f"index.html 已生成：{len(out):,} 字符 / {len(out.encode('utf-8')):,} 字节")
print(f"样本天数：{sum(s['n'] for s in slim['stocks']):,}")
if audit:
    print(f"稳健区间：{audit['q25']:06d} – {audit['q75']:06d}（单点重现率 {audit['hit_5k_pct']}%）")
