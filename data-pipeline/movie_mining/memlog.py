"""Peak memory per pipeline stage, including child processes (demucs, whisper-cli, ffmpeg).

    with peak_rss("afonya", "diarize"):
        ...
    # -> [mem] afonya diarize: peak 3.42 GB RSS (process + children), 212 s

A background thread samples RSS every 0.5 s, so very short spikes can be missed.
Without psutil it does nothing.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager


def _rss(proc) -> int:
    import psutil
    total = 0
    for p in [proc] + proc.children(recursive=True):
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


@contextmanager
def peak_rss(film: str, stage: str, interval: float = 0.5):
    try:
        import psutil
    except ImportError:
        yield
        return
    proc, peak, stop = psutil.Process(), [0], threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], _rss(proc))
            stop.wait(interval)

    t0 = time.time()
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        peak[0] = max(peak[0], _rss(proc))
        print(f"[mem] {film} {stage}: peak {peak[0] / 2**30:.2f} GB RSS (process + children), "
              f"{time.time() - t0:.0f} s", flush=True)
