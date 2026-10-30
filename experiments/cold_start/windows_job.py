"""Windows child ownership established while suspended, before any descendants."""

from __future__ import annotations

import ctypes
import os
import subprocess
import time
from ctypes import wintypes as w
from typing import Any


class BasicLimits(ctypes.Structure):
    _fields_ = [
        ("process_time", ctypes.c_longlong),
        ("job_time", ctypes.c_longlong),
        ("flags", w.DWORD),
        ("min_working_set", ctypes.c_size_t),
        ("max_working_set", ctypes.c_size_t),
        ("active_limit", w.DWORD),
        ("affinity", ctypes.c_size_t),
        ("priority", w.DWORD),
        ("scheduling", w.DWORD),
    ]


class IoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_ulonglong)
        for name in (
            "read_ops",
            "write_ops",
            "other_ops",
            "read_bytes",
            "write_bytes",
            "other_bytes",
        )
    ]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("basic", BasicLimits),
        ("io", IoCounters),
        ("process_memory", ctypes.c_size_t),
        ("job_memory", ctypes.c_size_t),
        ("peak_process_memory", ctypes.c_size_t),
        ("peak_job_memory", ctypes.c_size_t),
    ]


class Accounting(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_longlong)
        for name in ("user_time", "kernel_time", "period_user", "period_kernel")
    ]
    _fields_ += [
        (name, w.DWORD)
        for name in ("page_faults", "total_processes", "active_processes", "terminated_processes")
    ]


class ThreadEntry(ctypes.Structure):
    _fields_ = [
        ("size", w.DWORD),
        ("usage", w.DWORD),
        ("tid", w.DWORD),
        ("pid", w.DWORD),
        ("priority", w.LONG),
        ("delta", w.LONG),
        ("flags", w.DWORD),
    ]


class Job:
    def __init__(self) -> None:
        if os.name != "nt":
            raise RuntimeError("Physical supervisor requires native Windows")
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
            "SetInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
            "QueryInformationJobObject": (
                [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p],
                w.BOOL,
            ),
            "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
            "TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL),
            "CloseHandle": ([w.HANDLE], w.BOOL),
            "CreateToolhelp32Snapshot": ([w.DWORD, w.DWORD], w.HANDLE),
            "Thread32First": ([w.HANDLE, ctypes.POINTER(ThreadEntry)], w.BOOL),
            "Thread32Next": ([w.HANDLE, ctypes.POINTER(ThreadEntry)], w.BOOL),
            "OpenThread": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            "ResumeThread": ([w.HANDLE], w.DWORD),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = args, result
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def spawn(self, command: list[str], **kwargs: Any) -> subprocess.Popen:
        child = subprocess.Popen(command, creationflags=0x4 | 0x08000000, **kwargs)
        try:
            if not self.kernel.AssignProcessToJobObject(self.handle, int(child._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
            snapshot = self.kernel.CreateToolhelp32Snapshot(4, 0)
            if snapshot == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            resumed = 0
            try:
                entry = ThreadEntry()
                entry.size = ctypes.sizeof(entry)
                valid = self.kernel.Thread32First(snapshot, ctypes.byref(entry))
                while valid:
                    if entry.pid == child.pid:
                        thread = self.kernel.OpenThread(2, False, entry.tid)
                        if not thread:
                            raise ctypes.WinError(ctypes.get_last_error())
                        try:
                            if self.kernel.ResumeThread(thread) == 0xFFFFFFFF:
                                raise ctypes.WinError(ctypes.get_last_error())
                            resumed += 1
                        finally:
                            self.kernel.CloseHandle(thread)
                    valid = self.kernel.Thread32Next(snapshot, ctypes.byref(entry))
            finally:
                self.kernel.CloseHandle(snapshot)
            if resumed != 1:
                raise RuntimeError("Expected exactly one suspended primary thread")
            return child
        except BaseException:
            child.kill()
            child.wait(timeout=30)
            raise

    def accounting(self) -> dict[str, int]:
        result = Accounting()
        if not self.kernel.QueryInformationJobObject(
            self.handle, 1, ctypes.byref(result), ctypes.sizeof(result), None
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            name: int(getattr(result, name))
            for name in ("total_processes", "active_processes", "terminated_processes")
        }

    def terminate(self) -> None:
        if not self.kernel.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())
        deadline = time.monotonic() + 30
        while self.accounting()["active_processes"]:
            if time.monotonic() > deadline:
                raise TimeoutError("Windows job processes did not exit")
            time.sleep(0.01)

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
