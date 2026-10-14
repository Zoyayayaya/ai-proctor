#!/usr/bin/env python
"""
Platform helpers. Everything here must work on macOS, Linux AND Windows.

Why this module exists
======================
The first version of the live session used two Unix-only APIs:

    resource.getrusage(...).ru_maxrss     # no `resource` module on Windows
    os.statvfs(path)                      # does not exist on Windows

Both would have raised immediately on the Windows machine that ran the original
tests, so the whole pipeline would have failed before doing anything. Rather
than sprinkling `if sys.platform` checks through the scripts, the platform
differences are isolated here.

Also handles two Windows-specific practicalities:

  * camera capture is more reliable through the DirectShow backend
    (`cv2.CAP_DSHOW`) than the default MSMF one, which can be slow to open or
    silently fail on some webcams;
  * not every machine has the `mp4v` encoder available, so writer creation
    falls back through a short list of codecs instead of failing outright.
"""
from __future__ import annotations

import shutil
import sys

IS_WINDOWS = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"


# --------------------------------------------------------------------------
# resources
# --------------------------------------------------------------------------
def peak_rss_mb() -> float:
    """Peak resident memory of this process, in MB. 0.0 if unavailable."""
    if IS_WINDOWS:
        try:
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            c = _PMC()
            c.cb = ctypes.sizeof(c)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
            return c.PeakWorkingSetSize / (1024 * 1024) if ok else 0.0
        except Exception:
            return 0.0
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports kilobytes
        return r / (1024 * 1024) if IS_MAC else r / 1024
    except Exception:
        return 0.0


def free_gb(path) -> float:
    """Free disk space in GB at `path`. inf if it cannot be determined."""
    try:
        return shutil.disk_usage(str(path)).free / (1024 ** 3)
    except Exception:
        return float("inf")


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------
def open_camera(index: int, width: int, height: int, fps: float):
    """
    Open a webcam with platform-appropriate settings.

    On Windows the DirectShow backend is preferred: the default MSMF backend
    can take several seconds to open, or fail on webcams that work fine in
    every other application.
    """
    import cv2

    backends = []
    if IS_WINDOWS:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    cap = None
    for be in backends:
        c = cv2.VideoCapture(index, be)
        if c.isOpened():
            cap = c
            break
        c.release()
    if cap is None:
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def open_writer(path, fps: float, size):
    """
    Create a video writer, trying several codecs.

    `mp4v` is not guaranteed present on every build of OpenCV, and on Windows
    an unavailable encoder fails at write time rather than at creation, which
    is a confusing way to lose a recording. Try a short list and report which
    one worked.
    """
    import cv2

    for name in ("mp4v", "avc1", "MJPG"):
        try:
            w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*name), fps, size)
            if w.isOpened():
                return w, name
            w.release()
        except Exception:
            continue
    return None, None


# --------------------------------------------------------------------------
# device
# --------------------------------------------------------------------------
def pick_device(prefer: str = "auto") -> str:
    """
    Best available inference device.

    Order: explicit choice > CUDA (NVIDIA, typical on the Windows box that ran
    the original tests) > MPS (Apple) > CPU.
    """
    if prefer != "auto":
        return prefer
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def describe() -> str:
    return (f"platform={sys.platform} python={sys.version.split()[0]} "
            f"device={pick_device()}")


if __name__ == "__main__":
    print(describe())
    print(f"  free disk here : {free_gb('.') or 0:.1f} GB")
    print(f"  peak RSS       : {peak_rss_mb():.0f} MB")
    try:
        import cv2
        cap = open_camera(0, 640, 480, 15.0)
        if cap is None:
            print("  camera         : could NOT be opened")
            print("                   on Windows this is usually the privacy setting:")
            print("                   Settings > Privacy & security > Camera")
        else:
            ok, fr = cap.read()
            print(f"  camera         : OK, frame {None if not ok else fr.shape}")
            cap.release()
    except Exception as e:
        print(f"  camera         : check failed ({type(e).__name__})")
