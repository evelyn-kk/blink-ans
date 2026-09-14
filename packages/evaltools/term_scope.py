"""内容例外的举证工具：一次 `term_map.yaml` 改动到底影响了哪些查询。

背景：`AGENTS.md` §6 允许"内容改动且范围可证明有限"不跑检索验证集。R85 补
`覆盖索引`/`回表` 那组映射时，我拿"9 道无关提问都不触发"当证据——codex R90
（CR-104）指出那不构成证明：没触发的样本再多，也说不清触发面有多大。

举证因此固定为三件产出：**可枚举的资产与谓词 / 预期命中 / 碰撞负例**。

**这三件能说明什么、不能说明什么**（CR-110）：全部结论都限定在
**120 道已登记评测题这个固定观测集**上。它不是查询空间的样本，更不构成
"影响范围有限"的证明——把它说成证明是 R85 那次的老毛病换个说法。工具因此
还要求**每个改动键在观测集里至少有一条正向见证**：连落脚点都没有的键，
说明的是"不知道它影响什么"，按 §6 一律改跑检索验证集。

R91 的第一版有三条可绕过路径，codex R91 逐条复现（CR-106/107/108），
这一版按意见重写，三处都改成失败关闭：

1. **只看当前词典命中**（CR-106）→ 删除或遮蔽一条映射时，新词典下当然
   "零命中"，工具却报成"零影响"。现在**逐题比较旧/新 `expand_terms()` 的
   差分**：`added`（新增的检索词）与 `removed`（消失的检索词）任一非空就算
   受影响，删除映射因此和新增一样可见。
2. **`--expect ''` 通配一切**（CR-107）→ 前缀匹配下空串命中所有题，退出 0。
   现在只接受**精确题面**，并且要求**逐题声明完整的 added/removed**，与实测
   逐字比对；空串、重复、匹配不到题面一律拒绝。
3. **`--since HEAD~1` 会漂移**（CR-108）→ 与 CR-093 定下的"审计引用一律用
   固定 SHA"冲突。现在拒绝任何非 SHA 的写法，并把**解析后的完整 commit**
   写进产物。

`--expect-file` 的格式（YAML 或 JSON 皆可）::

    expect:
      - q: "已经建了包含查询所有列的覆盖索引，为什么执行时还是要回表访问堆，没有变快"
        added: ["covering index", "index-only scan", …]
        removed: []

用法::

    .venv/bin/python packages/evaltools/term_scope.py --since <完整或短 SHA> \
        --expect-file <预期文件> --negatives "…" "…" \
        --json bench/audits/<改动名>-scope.json

退出码：0 = 三件产出齐全且预期与实测逐字一致；1 = 举证不完整或对不上
（按 §6 这两种情况一律改跑检索验证集）。本工具**不判断**命中面大小是否
可接受——那是审查方的判断。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
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

_SHA = re.compile(r"^[0-9a-f]{7,40}$")


class ScopeError(ValueError):
    """举证本身不成立（引用会漂移、预期写法不合规…），与"范围大"是两回事。"""


def resolve_sha(ref: str) -> str:
    """只接受固定 SHA，返回完整 commit（CR-108）。

    `HEAD~1`、分支名、tag 都会随时间漂移——CR-093 已经为此定过规矩，
    工具层面必须失败关闭，否则产物里那句"相对 X 的差分"日后复现不出来。
    """
    if not _SHA.match(ref):
        raise ScopeError(
            f"--since 只接受固定 SHA，拿到的是 {ref!r}。"
            f"`HEAD~1` / 分支名 / tag 都会漂移（CR-093/CR-108），"
            f"请先 `git rev-parse --short <ref>` 取出具体值再传进来。"
        )
    out = subprocess.run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
                         capture_output=True, text=True, cwd=ROOT)
    if out.returncode != 0:
        raise ScopeError(f"--since {ref!r} 不是本仓库里的提交")
    return out.stdout.strip()


def _terms_at(sha: str | None) -> dict[str, list[str]]:
    if sha is None:
        raw = TERM_MAP.read_text(encoding="utf-8")
    else:
        raw = subprocess.run(["git", "show", f"{sha}:knowledge/term_map.yaml"],
                             capture_output=True, text=True, check=True, cwd=ROOT).stdout
    return (yaml.safe_load(raw) or {}).get("terms") or {}


def changed_keys(old: dict, new: dict) -> dict[str, tuple[list[str] | None, list[str] | None]]:
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


def _with_terms(terms: dict[str, list[str]], fn):
    """把某一版词典临时装进分词器，跑 `fn()`，然后**逐项还原**模块状态。

    还原清单必须覆盖 `_load()` 写的每一项——CR-109 就漏在 `_version` 上：
    它是 `dictionary_version()` 的来源，也是"索引词典版本与查询侧一致"那道
    护栏的依据（`store.ChunkStore`），停在临时 YAML 的版本上会让后面任何
    一次开索引报假的不一致。jieba 的自定义词是全局副作用，同样要清理。
    """
    from services.retrieval import tokenize as tk
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "term_map.yaml"
        path.write_text(yaml.safe_dump({"version": 1, "terms": terms},
                                       allow_unicode=True, sort_keys=False), encoding="utf-8")
        saved = {
            "path": tk.TERM_MAP_PATH, "ready": tk._ready, "version": tk._version,
            "map": dict(tk._term_map), "parts": dict(tk._KEY_PARTS),
        }
        # jieba 的自定义词是**全局**副作用。粗暴地"跑完 del_word"会把 jieba
        # 本来就认识的词一并删掉（实测：临时词典里放 `索引`，跑完之后
        # `tests/unit/test_tokenize.py` 里一条切词用例直接挂了）。
        # 因此按词逐个记住**进来之前的词频**，跑完原样放回：之前没有的才删。
        import jieba
        prior_freq = {zh: jieba.dt.FREQ.get(zh) for zh in terms if zh not in saved["map"]}
        try:
            tk.TERM_MAP_PATH = path
            tk._ready = False
            tk._term_map.clear()
            tk._KEY_PARTS.clear()
            return fn()
        finally:
            for zh, freq in prior_freq.items():
                if freq is None:
                    jieba.del_word(zh)
                else:
                    jieba.add_word(zh, freq=freq)
            tk.TERM_MAP_PATH = saved["path"]
            tk._term_map.clear(); tk._term_map.update(saved["map"])
            tk._KEY_PARTS.clear(); tk._KEY_PARTS.update(saved["parts"])
            tk._version = saved["version"]
            tk._ready = saved["ready"]


def _expansions_under(terms: dict[str, list[str]], questions: list[str]) -> dict[str, list[str]]:
    """某一版词典下每道题的展开词。

    必须真的换掉词典再算，不能只比键——`matched_terms()` 里有"最具体者胜"
    这类跨键规则（CR-013），删掉一个具体键会让通用键重新生效，只看键差分
    看不出这种**遮蔽**变化。
    """
    from services.retrieval import tokenize as tk
    return _with_terms(terms, lambda: {q: sorted(tk.expand_terms(q)) for q in questions})


def _matched_under(terms: dict[str, list[str]], questions: list[str]) -> dict[str, set[str]]:
    """某一版词典下每道题**命中了哪些键**（逐键归因用，见 key_witnesses）。"""
    from services.retrieval import tokenize as tk
    return _with_terms(terms, lambda: {q: set(tk.matched_terms(q)) for q in questions})


def expansion_diff(old_terms: dict, new_terms: dict,
                   questions: list[str]) -> dict[str, dict[str, list[str]]]:
    """逐题比较旧/新展开（CR-106）。返回 {题: {added: [...], removed: [...]}}，
    只含**有变化**的题。"""
    before = _expansions_under(old_terms, questions)
    after = _expansions_under(new_terms, questions)
    out: dict[str, dict[str, list[str]]] = {}
    for q in questions:
        added = sorted(set(after[q]) - set(before[q]))
        removed = sorted(set(before[q]) - set(after[q]))
        if added or removed:
            out[q] = {"added": added, "removed": removed}
    return out


def key_witnesses(old_terms: dict, new_terms: dict,
                  questions: list[str]) -> dict[str, list[str]]:
    """每个改动键的**正向见证**：观测集里哪些题的**命中键集合**因它而变（CR-110）。

    **按键归因，不按展开词归因**：几个键常常共享同一个英文词
    （`index-only scan` 同时出现在 `覆盖索引`/`回表`/`索引只扫描`/`仅索引扫描`
    的展开里），按词匹配会把一道题算成四个键的见证——第一版就是这么写的，
    等于给没有落脚点的键发了假证明。

    为什么按键要见证：整体"只有 1 道题受影响"说明不了**每一个**键的范围。
    没有见证不等于没有影响，等于**不知道**，按 §6 应当改跑检索验证集。
    """
    before = _matched_under(old_terms, questions)
    after = _matched_under(new_terms, questions)
    changed = changed_keys(old_terms, new_terms)
    out: dict[str, list[str]] = {}
    for key in changed:
        out[key] = sorted(q for q in questions
                          if (key in after[q]) != (key in before[q]))
    return out


def load_expectations(path: Path) -> list[dict]:
    """预期文件：逐题**精确题面** + 完整 added/removed（CR-107）。"""
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = spec.get("expect")
    if items is None:
        raise ScopeError(f"{path.name}: 缺 `expect:` 列表")
    seen = set()
    for item in items:
        q = (item or {}).get("q")
        if not isinstance(q, str) or not q.strip():
            raise ScopeError(f"{path.name}: 每条预期都要有非空的精确题面 `q`（不接受空串/前缀）")
        if q in seen:
            raise ScopeError(f"{path.name}: 题面重复: {q[:30]}")
        seen.add(q)
        for field in ("added", "removed"):
            if not isinstance(item.get(field, []), list):
                raise ScopeError(f"{path.name}: `{field}` 必须是列表")
    return items


def compare(expect: list[dict], actual: dict[str, dict[str, list[str]]],
            known_questions: set[str]) -> list[str]:
    """预期 vs 实测，逐题逐词比对；两个方向都判。"""
    problems: list[str] = []
    for item in expect:
        q = item["q"]
        if q not in known_questions:
            problems.append(f"预期里的题面不在任何已登记评测集中（写错或不够精确）: {q[:36]}")
            continue
        got = actual.get(q)
        if got is None:
            problems.append(f"声明了预期，实测该题展开没有变化: {q[:36]}")
            continue
        for field in ("added", "removed"):
            want, have = sorted(item.get(field, [])), got[field]
            if want != have:
                problems.append(
                    f"{q[:28]} 的 {field} 对不上：预期 {want} / 实测 {have}")
    for q in sorted(set(actual) - {i["q"] for i in expect}):
        problems.append(f"实测受影响但没写进预期: {q[:36]}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help="固定 SHA（不接受 HEAD~1 等会漂移的写法）")
    ap.add_argument("--expect-file", help="预期文件：逐题精确题面 + 完整 added/removed")
    ap.add_argument("--negatives", nargs="*", default=[],
                    help="碰撞负例：含该键组成字但语义无关的提问，其展开必须不变")
    ap.add_argument("--json", help="产物路径，供存档与复核")
    args = ap.parse_args()

    try:
        sha = resolve_sha(args.since)
    except ScopeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    old_terms, new_terms = _terms_at(sha), _terms_at(None)
    changed = changed_keys(old_terms, new_terms)
    labelled = all_eval_questions()
    questions = [q for _, q in labelled]
    label_of = {q: label for label, q in labelled}

    print(f"① 可枚举的资产（相对 {sha[:12]}，完整 commit {sha}）：{len(changed)} 个键")
    for k, (old, new) in changed.items():
        state = "新增" if old is None else ("删除" if new is None else "改值")
        print(f"   [{state}] {k}: {old} -> {new}")
    print("\n   匹配谓词（见 tokenize.matched_terms）：子串命中，或键的组成词全部出现；"
          "命中后按'最具体者胜'丢弃被包含的键。")

    actual = expansion_diff(old_terms, new_terms, questions)
    print(f"\n② 展开差分（逐题比较旧/新 expand_terms，共 {len(questions)} 道题）："
          f"{len(actual)} 道受影响")
    for q, d in actual.items():
        print(f"   [{label_of[q]:<10}] +{d['added']} -{d['removed']} :: {q[:40]}")

    witnesses = key_witnesses(old_terms, new_terms, questions)
    unwitnessed = [k for k, qs in witnesses.items() if not qs]
    print(f"\n   逐键正向见证（观测集 {len(questions)} 道题里能归因到该键的）：")
    for key, qs in witnesses.items():
        print(f"     {'✓' if qs else '✗'} {key}: {len(qs)} 道" +
              (f" 例：{qs[0][:32]}" if qs else "  ← 观测集里没有落脚点"))

    problems: list[str] = []
    if args.expect_file:
        try:
            expect = load_expectations(Path(args.expect_file))
        except ScopeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        problems = compare(expect, actual, set(questions))
        if problems:
            print("\n   ✗ 预期与实测不一致：")
            for m in problems:
                print("     -", m)
        else:
            print(f"\n   ✓ 预期与实测逐词一致（声明 {len(expect)} 题）")
    else:
        print("\n   ⚠ 没给 --expect-file：预期这一步没做")

    print(f"\n③ 碰撞负例（{len(args.negatives)} 条，展开必须不变）：")
    neg_diff = expansion_diff(old_terms, new_terms, args.negatives) if args.negatives else {}
    for q in args.negatives:
        d = neg_diff.get(q)
        print(f"   {'✗ 展开变了' if d else '✓ 不受影响'} {d if d else ''} :: {q[:48]}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "since_resolved": sha,
            "changed_keys": {k: {"old": v[0], "new": v[1]} for k, v in changed.items()},
            "affected": actual, "key_witnesses": witnesses,
             "unwitnessed_keys": unwitnessed,
             "expect_file": args.expect_file,
            "expect_problems": problems,
            "negatives": args.negatives,
            "negatives_affected": neg_diff,
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    if problems:
        print("\n预期与实测对不上：要么范围判断错了，要么预期写得不准，按 §6 改跑检索验证集。")
        return 1
    if neg_diff:
        print("\n有碰撞负例的展开发生变化：范围不是你以为的那个，按 §6 改跑检索验证集。")
        return 1
    if not args.expect_file:
        print("\n没有事先声明预期（--expect-file）：举证不完整。")
        return 1
    if not args.negatives:
        print("\n没有给碰撞负例：举证不完整（『没触发的样本』不等于『范围有限』）。")
        return 1
    if unwitnessed:
        print(f"\n这些键在观测集里没有正向见证：{unwitnessed}。"
              f"\n观测集是**固定的 {len(questions)} 道题**，不是查询空间——"
              f"没有见证说明的是『不知道它影响什么』，不是『它没有影响』。"
              f"按 §6 应当改跑检索验证集。")
        return 1
    print("\n三件产出齐了。命中面是否可接受由审查方判断，本工具不下这个结论。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
