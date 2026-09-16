"""内存内 PCM 分片转写的最小契约。

浏览器在 I4 使用 Web Audio 将音频重采样为 16 kHz、单声道、little-endian
signed PCM；这里因此不接受文件路径、更不创建临时文件。每次 partial 转写都在
同一段内存波形上运行，最终事件产生后立即清空原始 PCM。
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Callable, Protocol

import numpy as np


SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
# Whisper 的一个处理窗口也是 30 秒；把入口上限钉在这里，单会话不会因为客户端
# 永不 final 而无限积压内存。超过上限必须让客户端开始新段，而不是静默截断语音。
MAX_SEGMENT_SECONDS = 30
MAX_PCM_BYTES = SAMPLE_RATE * BYTES_PER_SAMPLE * MAX_SEGMENT_SECONDS


class Transcriber(Protocol):
    def __call__(self, waveform: np.ndarray, *, language: str) -> str: ...


class TranscriptionFailed(RuntimeError):
    """本地转写器已被调用但未产生文本；不是客户端 PCM 格式错误。"""


@dataclass(frozen=True)
class TranscriptEvent:
    text: str
    final: bool
    sequence: int

    def as_sse_event(self) -> dict:
        # 不把 samples/PCM 放进事件：SSE 可能被代理缓存或日志采集，文本已经是
        # 最小必要输出，原始音频绝不能沿事件链泄漏。
        return {
            "type": "transcript",
            "text": self.text,
            "final": self.final,
            "sequence": self.sequence,
        }


class TranscriptSession:
    """单个发言段的顺序追加/转写状态，原始音频仅存在于 `_pcm`。"""

    def __init__(self, language: str, transcribe: Transcriber) -> None:
        if language not in {"zh", "en"}:
            raise ValueError(f"unsupported ASR language: {language!r}")
        self.language = language
        self._transcribe = transcribe
        self._pcm = bytearray()
        self._sequence = 0
        self._finished = False
        self._lock = RLock()

    @property
    def buffered_pcm_bytes(self) -> int:
        with self._lock:
            return len(self._pcm)

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._finished

    def append(self, pcm_s16le: bytes, *, final: bool = False) -> TranscriptEvent:
        """追加一段 16 kHz mono PCM 并同步取得最新转写。

        持锁跨越本地模型调用是有意的：同一 utterance 的 partial/final 不允许
        乱序完成并覆盖。网关应把此同步调用放入工作线程，不能阻塞事件循环。
        """
        if not pcm_s16le:
            raise ValueError("PCM chunk must not be empty")
        if len(pcm_s16le) % BYTES_PER_SAMPLE:
            raise ValueError("PCM chunk must contain whole 16-bit samples")
        with self._lock:
            if self._finished:
                raise RuntimeError("transcript session is already final")
            if len(self._pcm) + len(pcm_s16le) > MAX_PCM_BYTES:
                raise ValueError(
                    f"PCM segment exceeds {MAX_SEGMENT_SECONDS}s / {MAX_PCM_BYTES} bytes"
                )
            self._pcm.extend(pcm_s16le)
            # copy() 让 numpy 不再引用 bytearray；最终 clear 后模型调用结果与输入
            # 不会共享可变内存。
            waveform = np.frombuffer(bytes(self._pcm), dtype="<i2").astype(np.float32)
            waveform /= 32768.0
            # final 是语音段的不可重试边界：即使本地模型加载/推理失败，也不能
            # 留住用户原始 PCM 等客户端再 append。partial 的异常则保留缓冲，允许
            # 网络抖动或临时模型故障后由客户端重试同一段。
            try:
                text = self._transcribe(waveform, language=self.language).strip()
            except Exception as exc:
                raise TranscriptionFailed("local transcription failed") from exc
            finally:
                if final:
                    self._pcm.clear()
                    self._finished = True
            self._sequence += 1
            event = TranscriptEvent(text=text, final=final, sequence=self._sequence)
            return event

    def cancel(self) -> None:
        """断开或用户取消时无条件释放尚未 final 的音频。"""
        with self._lock:
            self._pcm.clear()
            self._finished = True


def mlx_whisper_transcriber(
    waveform: np.ndarray,
    *,
    language: str,
    model: str = "mlx-community/whisper-large-v3-turbo",
) -> str:
    """真实本地 adapter；延迟导入以免网关启动额外占用 M4 内存。"""
    from mlx_whisper import transcribe

    result = transcribe(
        waveform,
        path_or_hf_repo=model,
        language=language,
        initial_prompt=None,
        verbose=None,
    )
    text = result.get("text")
    if not isinstance(text, str):
        raise RuntimeError("mlx-whisper returned no text")
    return text
