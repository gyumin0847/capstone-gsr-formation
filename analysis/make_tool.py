"""stats.json -> self-contained feedback tool HTML.
usage: python analysis/make_tool.py stats.json web/template.html formation_tool.html"""
import json, sys
stats, tpl, out = sys.argv[1:4]
data = json.dumps(json.load(open(stats, encoding='utf-8')), ensure_ascii=False, separators=(',', ':'))
html = open(tpl, encoding='utf-8').read().replace('__DATA__', data.replace('</', '<\\/'))
open(out, 'w', encoding='utf-8').write(html)
print('written:', out)
