"""判别性实验：把归因脚本逐处改坏，看哪些回归会失败（T-023 R187）。

`bench/bench_language_failure_attribution.py` 是本轮**新增**的文件，因此
"在上一个 commit 上跑这些测试会失败"只能证明模块不存在，不能证明判据有判别性
（§5.2 明确把这种情况算作判别性弱，必须如实说明）。这个脚本给出另一种证据：
保持测试不变，只把实现改坏一处，看对应的那条测试会不会失败。

每处变异都是**精确的 old→new 全字符串替换**，原样记录在产物里，
不是自然语言描述的"大意如此"（CR-105：报告"X 条失败"须保存精确 patch）。

用法：.venv/bin/python bench/mutate_language_failure_attribution.py \
        --json bench/audits/t023-r187-mutation-discriminativeness.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "bench" / "bench_language_failure_attribution.py"
TESTS = ROOT / "tests" / "unit" / "test_language_failure_attribution.py"

MUTATIONS = [
    {
        "name": "身份字段不同时照样放行",
        "old": "        if zh_value != en_value:",
        "new": "        if False:",
    },
    {
        "name": "身份字段两边都缺时当成相等",
        "old": "        if zh_value is None or en_value is None:",
        "new": "        if False:",
    },
    {
        "name": "两侧词项从未同时出现时报成最大窗口，而不是 None",
        "old": "        if re.search(_PROXIMITY.sub(\"{0,%d}\" % n, keypoint), text, re.S):\n"
               "            return n\n    return None",
        "new": "        if re.search(_PROXIMITY.sub(\"{0,%d}\" % n, keypoint), text, re.S):\n"
               "            return n\n    return _MAX_WINDOW",
    },
    {
        "name": "证据正文哈希不再核对",
        "old": "        if digest != item[\"text_sha256\"]:",
        "new": "        if False:",
    },
    {
        "name": "拒答也算「未标引用」失败",
        "old": "    if case[\"expect\"] == \"answered\" and not case[\"declined\"] and case[\"cited\"] == 0:",
        "new": "    if case[\"cited\"] == 0:",
    },
    {
        "name": "语料覆盖改成只匹配块首，覆盖统计被压成 0",
        "old": "            row[\"id\"] for row in store.execute(\"SELECT id, text FROM chunks\")\n"
               "            if compiled.search(row[\"text\"])",
        "new": "            row[\"id\"] for row in store.execute(\"SELECT id, text FROM chunks\")\n"
               "            if compiled.match(row[\"text\"])",
    },
    {
        "name": "「证据里有没有」改看答案文本，两件事被混为一谈",
        "old": "            arm: sorted(cid for cid, text in texts[arm].items() if pattern.search(text))",
        "new": "            arm: sorted(cid for cid, text in texts[arm].items()\n"
               "                        if pattern.search({\"zh\": zh_case, \"en\": en_case}[arm][\"answer_text\"] or \"\"))",
    },
]

_FAILED = re.compile(r"^FAILED (\S+)", re.M)


def run(module_path: Path) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(TESTS), "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT, text=True, capture_output=True, check=False,
        env={**dict(__import__("os").environ), "ATTRIBUTION_MODULE": str(module_path)},
    )
    failed = sorted({m.split("::")[-1] for m in _FAILED.findall(result.stdout)})
    return {"exit_code": result.returncode, "failed_tests": failed,
            "summary": result.stdout.strip().splitlines()[-1] if result.stdout else ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    source = TARGET.read_text(encoding="utf-8")
    baseline = run(TARGET)
    if baseline["failed_tests"]:
        raise SystemExit(f"未变异时就有测试失败，先修好：{baseline['failed_tests']}")

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for mutation in MUTATIONS:
            if source.count(mutation["old"]) != 1:
                raise SystemExit(
                    f"变异锚点在源文件里出现 {source.count(mutation['old'])} 次，"
                    f"不是唯一：{mutation['name']}"
                )
            path = Path(tmp) / "mutated.py"
            path.write_text(source.replace(mutation["old"], mutation["new"]), encoding="utf-8")
            outcome = run(path)
            rows.append({**mutation, **outcome})

    out = {
        "round": "R187",
        "target": str(TARGET.relative_to(ROOT)),
        "tests": str(TESTS.relative_to(ROOT)),
        "note": "脚本为本轮新增，旧 commit 上测试只会因模块不存在而失败；"
                "这份产物给的是「实现改坏一处则对应回归失败」的判别性证据。",
        "baseline": baseline,
        "mutations": rows,
        "mutations_with_no_failing_test": [r["name"] for r in rows if not r["failed_tests"]],
    }
    path = Path(args.json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        print(f"{'✓' if row['failed_tests'] else '✗'} {row['name']}：{row['failed_tests'] or '无测试失败'}")
    print(f"写入 {path}")
    return 1 if out["mutations_with_no_failing_test"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
