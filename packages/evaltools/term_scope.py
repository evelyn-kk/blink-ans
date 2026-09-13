"""内容例外的举证工具：一次 `term_map.yaml` 改动到底影响了哪些查询（CR-104）。

背景：`AGENTS.md` §6 允许"内容改动且范围可证明有限"不跑检索验证集。R85 补
`覆盖索引`/`回表` 那组映射时，我拿"9 道无关提问都不触发"当证据——codex R90
指出那不构成证明：**没触发的样本再多，也说不清触发面有多大**。

这个工具把举证变成三件可核对的产出：

1. **可枚举的资产与谓词**：这次改了哪些键、每个键的匹配谓词是什么
   （子串命中，或"组成词全部出现"——见 `tokenize.matched_terms`）；
2. **预期命中**：在全部已登记评测题上，哪些题会因这次改动而改变展开；
3. **碰撞负例**：包含该键**组成字**但语义无关的提问，必须**不**触发——
   这才是"范围有限"的反面证据（例如 `堆` 与"消息堆积"、`回表` 与"回滚表"）。

用法：
    .venv/bin/python packages/evaltools/term_scope.py --since HEAD~1
    .venv/bin/python packages/evaltools/term_scope.py --since 2cbe470 --json out.json

退出码：0 = 三件产出都齐了；1 = 有键拿不出碰撞负例（举证不完整，按 §6
应当改跑检索验证集）。它**不判断**命中面大小是否可接受——那是审查方的判断。
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402

TERM_MAP = ROOT / "knowledge" / "term_map.yaml"
EVAL_FILES = (
    ("basic", ROOT / "knowledge/eval/basic_questions.yaml", "questions"),
    ("scenario", ROOT / "knowledge/eval/scenario_questions.yaml", "questions"),
    ("probe", ROOT / "knowledge/eval/ranking_probe.yaml", "probes"),
    ("validation", ROOT / "knowledge/eval/ranking_validation.yaml", "cases"),
)


def _terms_at(ref: str | None) -> dict[str, list[str]]:
    if ref is None:
        raw = TERM_MAP.read_text(encoding="utf-8")
    else:
        raw = subprocess.run(
            ["git", "show", f"{ref}:knowledge/term_map.yaml"],
            capture_output=True, text=True, check=True, cwd=ROOT,
        ).stdout
    return (yaml.safe_load(raw) or {}).get("terms") or {}


def changed_keys(since: str) -> dict[str, tuple[list[str] | None, list[str] | None]]:
    """返回 {键: (旧展开, 新展开)}；新增键旧值为 None，删除键新值为 None。"""
    old, new = _terms_at(since), _terms_at(None)
    keys = sorted(set(old) | set(new))
    return {k: (old.get(k), new.get(k)) for k in keys if old.get(k) != new.get(k)}


def all_eval_questions() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for label, path, key in EVAL_FILES:
        if not path.exists():
            continue
        for item in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get(key, []):
            q = item.get("q")
            if q:
                out.append((label, q))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help="与哪个提交对比（固定 SHA，CR-093）")
    ap.add_argument("--json", help="把结果写到文件，便于贴进 wait-for-review.md")
    ap.add_argument("--negatives", nargs="*", default=[],
                    help="碰撞负例：包含该键组成字但语义无关的提问，必须不触发")
    ap.add_argument("--expect", nargs="*", default=None,
                    help="**事先声明**的预期命中（题面前缀，可多条）。给了就逐条比对："
                         "漏命中或多命中都判失败。不给只报实测，并提示这一步没做")
    args = ap.parse_args()

    from services.retrieval import tokenize as tk
    importlib.reload(tk)

    changed = changed_keys(args.since)
    if not changed:
        print(f"与 {args.since} 相比，term_map.yaml 没有键级改动。")
        return 0

    print(f"① 可枚举的资产（相对 {args.since}）：{len(changed)} 个键")
    for k, (old, new) in changed.items():
        state = "新增" if old is None else ("删除" if new is None else "改值")
        print(f"   [{state}] {k} -> {new}")
    print("\n   匹配谓词（见 tokenize.matched_terms）：子串命中，或键的组成词"
          "全部出现在提问里；命中后按'最具体者胜'丢弃被包含的键。")

    changed_set = set(changed)
    print(f"\n② 预期命中（全部已登记评测题 {len(all_eval_questions())} 道）：")
    hits = []
    for label, q in all_eval_questions():
        fired = sorted(changed_set & set(tk.matched_terms(q)))
        if fired:
            hits.append({"set": label, "q": q, "fired": fired})
            print(f"   [{label:<10}] {fired} :: {q[:46]}")
    if not hits:
        print("   （无）——注意：这本身不是'范围有限'的证据，只说明评测集没覆盖到")

    # 先声明预期、再跑、再比差异（否则"预期命中"退化成"事后解释实测结果"）
    expect_problems: list[str] = []
    if args.expect is None:
        print("\n   ⚠ 没有用 --expect 事先声明预期命中：这一步没做，"
              "本次只是把实测结果列出来，不构成'预期与实测一致'的证据")
    else:
        actual = {h["q"] for h in hits}
        matched = set()
        for want in args.expect:
            got = [q for q in actual if q.startswith(want)]
            if not got:
                expect_problems.append(f"声明了预期但没命中: {want[:40]}")
            matched.update(got)
        for q in sorted(actual - matched):
            expect_problems.append(f"命中了但没在预期里: {q[:40]}")
        if expect_problems:
            print("\n   ✗ 预期与实测不一致：")
            for m in expect_problems:
                print("     -", m)
        else:
            print(f"\n   ✓ 预期与实测一致（声明 {len(args.expect)} 条前缀，"
                  f"覆盖 {len(actual)} 道题）")

    print(f"\n③ 碰撞负例（{len(args.negatives)} 条，必须都不触发）：")
    leaked = []
    for q in args.negatives:
        fired = sorted(changed_set & set(tk.matched_terms(q)))
        mark = "✗ 触发了" if fired else "✓ 不触发"
        print(f"   {mark} {fired if fired else ''} :: {q[:52]}")
        if fired:
            leaked.append({"q": q, "fired": fired})

    missing_neg = not args.negatives
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"since": args.since, "changed": {k: v[1] for k, v in changed.items()},
             "expected_hits": hits, "declared_expect": args.expect,
             "expect_problems": expect_problems,
             "negatives": args.negatives, "leaked": leaked},
            ensure_ascii=False, indent=1), encoding="utf-8")

    if expect_problems:
        print("\n预期命中与实测对不上：要么范围判断错了，要么预期写得不准，"
              "两种都按 §6 改跑检索验证集。")
        return 1
    if leaked:
        print("\n有碰撞负例被触发：范围不是你以为的那个，按 §6 应当改跑检索验证集。")
        return 1
    if args.expect is None:
        print("\n没有事先声明预期命中（--expect）：举证不完整。")
        return 1
    if missing_neg:
        print("\n没有给碰撞负例：举证不完整（『没触发的样本』不等于『范围有限』），"
              "按 §6 应当补负例或改跑检索验证集。")
        return 1
    print("\n三件产出齐了。命中面是否可接受由审查方判断，本工具不下这个结论。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
