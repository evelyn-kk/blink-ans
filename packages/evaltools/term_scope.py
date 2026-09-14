"""内容例外的举证工具：一次 `term_map.yaml` 改动到底影响了哪些查询。

背景：`AGENTS.md` §6 允许"内容改动且范围可证明有限"不跑检索验证集。R85 补
`覆盖索引`/`回表` 那组映射时，我拿"9 道无关提问都不触发"当证据——codex R90
（CR-104）指出那不构成证明：没触发的样本再多，也说不清触发面有多大。

举证因此固定为三件产出：**可枚举的资产与谓词 / 预期命中 / 碰撞负例**。

**这三件能说明什么、不能说明什么**（CR-110）：全部结论都限定在
**已登记评测题这个固定观测集**上（120 条登记题面，去重后 106 道）。它不是查询空间的样本，更不构成
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

R93 的版本在本进程里换词典再还原，codex R93 指出这个差分**根本没有隔离**
（CR-111）：`_load()` 会把每个键 `jieba.add_word`，而 jieba 的前缀树是进程级
全局状态。加载当前词典之后再以旧词典运行，当前独有的键仍在 jieba 里——
旧态被新词污染。现在**两版词典各自在短生命周期子进程里算**
（`_scope_worker.py`），主进程全程不碰分词器，还原清单随之取消。

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


WORKER = Path(__file__).with_name("_scope_worker.py")

# 每版词典的实测结果缓存：键是词典本身（同一版词典的结果与算过哪些题无关），
# 值里按题累积。缓存只保证"同一批题不重算"，**不会**把两批题合成一次子进程——
# CR-112 就是栽在这上面：我据此写了"四次调用只落成两次子进程"，实际是 4 次，
# 因为负例是另一批题，两版词典各自又起了一个 worker。要真的压到两次，得在
# 算之前**主动把观测集与负例并成一批**喂进来（见 main 里的 _prime）。
_CACHE: dict[str, dict[str, dict[str, list[str]]]] = {}


def _under(terms: dict[str, list[str]], questions: list[str]) -> dict[str, dict[str, list[str]]]:
    """某一版词典下每道题的展开词与命中键，**在短生命周期子进程里算**（CR-111）。

    上一版是在本进程里换词典再还原。还原清单能管住 `tokenize` 的模块状态
    （CR-109 补的那五项），却管不住 jieba：`_load()` 对每个键 `add_word`，
    改的是第三方库的全局前缀树。于是加载当前词典之后再以旧词典运行时，
    **当前独有的键仍然在 jieba 里**（实测 `索引只扫描` 词频 900，120 道题里
    19 道的切词与干净进程不同）——旧态被新词污染，差分随之漏报。

    快照"进来之前的词频"救不了这个：它只覆盖我们**知道要记**的词，而这里
    要排除的恰恰是"上一版留下的、清单里没有的"那些。换成进程边界之后，
    主进程全程不碰分词器状态，也就没有还原清单要维护了。
    """
    key = json.dumps(terms, ensure_ascii=False, sort_keys=True)
    cached = _CACHE.setdefault(key, {"expansions": {}, "matched": {}})
    # 去重：负例里出现一道已登记题面是合法的（测试里就有），不该让 worker 算两遍。
    missing = list(dict.fromkeys(q for q in questions if q not in cached["expansions"]))
    if missing:
        with tempfile.TemporaryDirectory() as d:
            inp, outp = Path(d) / "in.json", Path(d) / "out.json"
            inp.write_text(json.dumps({"terms": terms, "questions": missing},
                                      ensure_ascii=False), encoding="utf-8")
            run = subprocess.run([sys.executable, str(WORKER), str(inp), str(outp)],
                                 capture_output=True, text=True, cwd=ROOT)
            if run.returncode != 0 or not outp.exists():
                raise ScopeError(
                    f"隔离子进程算词典结果失败（退出码 {run.returncode}）：\n"
                    + (run.stderr or "")[-800:])
            got = json.loads(outp.read_text(encoding="utf-8"))
        cached["expansions"].update(got["expansions"])
        cached["matched"].update(got["matched"])
    return cached


def _expansions_under(terms: dict[str, list[str]], questions: list[str]) -> dict[str, list[str]]:
    """某一版词典下每道题的展开词。

    必须真的换掉词典再算，不能只比键——`matched_terms()` 里有"最具体者胜"
    这类跨键规则（CR-013），删掉一个具体键会让通用键重新生效，只看键差分
    看不出这种**遮蔽**变化。
    """
    got = _under(terms, questions)["expansions"]
    return {q: got[q] for q in questions}


def _matched_under(terms: dict[str, list[str]], questions: list[str]) -> dict[str, set[str]]:
    """某一版词典下每道题**命中了哪些键**（逐键归因用，见 key_witnesses）。"""
    got = _under(terms, questions)["matched"]
    return {q: set(got[q]) for q in questions}


def _prime(versions: list[dict[str, list[str]]], *batches: list[str]) -> None:
    """把所有要算的题**并成一批**先喂给每版词典，一版一个子进程（CR-112）。

    不做这一步的话，观测集与负例是两次独立调用，缓存只能各自命中各自那批，
    两版词典就会各起两个 worker。起一次 worker（起 Python + 载 jieba 词典）实测
    0.26~0.29 秒（空闲，n=12），是这里唯一值钱的东西，所以要按"版本数"收费而不是
    "调用次数"。**这个数是微基准，不是固定成本**（CR-113）：后台跑着快速门禁时
    同样量法是 0.28~0.32 秒，换台机器只会差更多。
    """
    every = [q for batch in batches for q in batch]
    if not every:
        return
    for terms in versions:
        _under(terms, every)


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
    # 登记的题面里有重复（有 14 道同时登记在两个评测集里）。不去重的话观测面会
    # 被夸大，见证列表还会把同一道题列两遍——R93 存档的 `term-map-r85-scope.json`
    # 里 `回表`/`覆盖索引` 各"2 道见证"，其实是同一道题数了两次。
    questions: list[str] = []
    label_of: dict[str, str] = {}
    for label, q in labelled:
        if q not in label_of:
            label_of[q] = label
            questions.append(q)

    print(f"① 可枚举的资产（相对 {sha[:12]}，完整 commit {sha}）：{len(changed)} 个键")
    for k, (old, new) in changed.items():
        state = "新增" if old is None else ("删除" if new is None else "改值")
        print(f"   [{state}] {k}: {old} -> {new}")
    print("\n   匹配谓词（见 tokenize.matched_terms）：子串命中，或键的组成词全部出现；"
          "命中后按'最具体者胜'丢弃被包含的键。")

    _prime([old_terms, new_terms], questions, args.negatives)

    actual = expansion_diff(old_terms, new_terms, questions)
    print(f"\n② 展开差分（逐题比较旧/新 expand_terms，登记 {len(labelled)} 条题面、"
          f"去重后 {len(questions)} 道）：{len(actual)} 道受影响")
    for q, d in actual.items():
        print(f"   [{label_of[q]:<10}] +{d['added']} -{d['removed']} :: {q[:40]}")

    witnesses = key_witnesses(old_terms, new_terms, questions)
    unwitnessed = [k for k, qs in witnesses.items() if not qs]
    print(f"\n   逐键正向见证（观测集去重后 {len(questions)} 道题里能归因到该键的）：")
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
