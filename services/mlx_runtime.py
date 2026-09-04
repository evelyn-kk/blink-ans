"""进程级 MLX 原生扩展可用性哨兵（CR-036）。

`mlx_lm` 与 `mlx_embeddings` 底层共用同一份 `mlx.core` 原生扩展（nanobind
绑定 C++ 类型）。如果第一次 `import` 在原生扩展的 `PyInit_` 执行到一半时失败
（例如没有 Metal 设备），C++ 层的静态类型注册表会留下一部分已经注册、但
Python 侧模块对象没有成功产出的状态——这是**进程级、不受 Python
try/except 或对象生命周期约束**的状态：`sys.modules` 里失败的模块会被移除，
但 C++ 静态注册表不会跟着回滚。同一个进程里再尝试导入同一份原生扩展
（哪怕是通过完全不同的 Python 包，比如 `InferenceEngine` 用的 `mlx_lm` 已经
失败之后，`Embedder` 才去导入 `mlx_embeddings`），nanobind 发现同一个 C++
类型被第二次注册时会判定为不可恢复的状态损坏，直接调用它的 fatal error
handler——这不是一个 Python 异常，`try/except Exception` 完全拦不住，
会直接中止整个进程（CR-036）。

因此第一次导入失败之后，这个进程剩下的生命周期里都不能再尝试导入 mlx
原生扩展，不管是通过哪个 Python 包。`broken` 这个标志一旦置位就**不会**被
清除——这一点和 `InferenceEngine.status.error`/`Embedder.error` 不同（那两个
描述的是"最近一次加载尝试"的结果，可以在重试成功后清空，见 CR-037）：
这里描述的是这个操作系统进程里原生扩展的物理状态，进程不重启就回不去，
"重试"这个概念对它不成立。
"""

from __future__ import annotations

broken: bool = False
broken_reason: str | None = None


def mark_broken(reason: str) -> None:
    """记录"这个进程的 mlx 原生扩展已经处于不可信状态"，且只记一次。

    只在真正的导入失败（`ImportError`/`ModuleNotFoundError`）时调用——
    模型加载阶段的其他失败（网络、权重文件缺失等）发生在原生扩展已经
    成功导入之后，不构成这里描述的风险，不应该调用这个函数。
    """
    global broken, broken_reason
    if not broken:
        broken = True
        broken_reason = reason
