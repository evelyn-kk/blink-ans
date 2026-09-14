"""短生命周期子进程：在**干净的分词器里**算某一版词典下的展开与命中。

必须是独立进程（CR-111）。`tokenize._load()` 会对每个映射键调
`jieba.add_word(zh, freq=900)`，那是**第三方库的进程级前缀树**。同一个进程
里先后装两版词典，后装那版的新词会留在 jieba 里改变先装那版的切词——于是
"旧词典下的结果"其实是"旧词典 + 新词"的结果，差分**漏报**。

实测（R94）：加载当前词典后再以 R85 旧词典运行，当前独有的 `索引只扫描`
等键仍在 `jieba.dt.FREQ` 里（词频 900），120 道题里有 19 道的 `tokenize()`
输出与干净进程不同。

进程边界是唯一能让两版词典互不可见的手段：进程内的"还原清单"做不到，因为
`add_word` 改的是别人的全局状态，而"进来之前是什么"这种快照只能覆盖我们
知道要记的那些词（CR-109 已经在这条路上摔过一次）。

协议：`argv[1]` 是入参 JSON（`{"terms": …, "questions": [...]}`），
`argv[2]` 是出参 JSON 的写入路径。**不走 stdout**——jieba 启动时会往
stderr/stdout 打日志，用文件交换省得去猜哪一行才是结果。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402


def main() -> int:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    terms: dict[str, list[str]] = payload["terms"]
    questions: list[str] = payload["questions"]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "term_map.yaml"
        path.write_text(yaml.safe_dump({"version": 1, "terms": terms},
                                       allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
        from services.retrieval import tokenize as tk

        # 本进程用完即弃，因此**不需要**还原任何东西：改完就退出，
        # 污染随进程一起消失。这正是把这段搬出主进程的理由。
        tk.TERM_MAP_PATH = path
        result = {
            "expansions": {q: sorted(tk.expand_terms(q)) for q in questions},
            "matched": {q: sorted(tk.matched_terms(q)) for q in questions},
        }
    Path(sys.argv[2]).write_text(json.dumps(result, ensure_ascii=False),
                                 encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
