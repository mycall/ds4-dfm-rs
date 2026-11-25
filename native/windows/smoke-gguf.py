"""Run one N1X GGUF generation and stop it if Windows runs short of RAM."""
import argparse
import ctypes
from ctypes import wintypes
from pathlib import Path
import subprocess
import sys
import tempfile
import time


class MemoryStatus(ctypes.Structure):
    _fields_ = [("size", wintypes.DWORD), ("load", wintypes.DWORD)] + [
        (name, ctypes.c_ulonglong)
        for name in ("total", "available", "commit", "commit_available",
                     "virtual", "virtual_available", "extended")
    ]


def available():
    status = MemoryStatus()
    status.size = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return min(status.available, status.commit_available)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exe", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    reserve = 3 * 1024**3
    low = available()
    if low <= reserve:
        raise RuntimeError("fewer than 3 GiB available before launch")
    command = [str(args.exe.resolve()), "-m", str(args.model.resolve()), "--backend", args.backend,
               "--ctx", "512", "--threads", "12", "--temp", "0", "--nothink",
               "-p", "What is two plus two? Reply with just the digit.", "-n", "8"]
    start = time.monotonic()
    with tempfile.TemporaryFile() as answer:
        proc = subprocess.Popen(command, stdout=answer)
        try:
            while proc.poll() is None:
                low = min(low, available())
                if low < reserve:
                    raise RuntimeError("3 GiB Windows memory reserve crossed")
                time.sleep(0.1)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        answer.seek(0)
        output = answer.read().decode("utf-8").strip()
    print(f"{args.backend}: {output!r}, {time.monotonic()-start:.1f}s, "
          f"lowest available Windows memory {low/1024**3:.2f} GiB", flush=True)
    if proc.returncode or output != "4":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
