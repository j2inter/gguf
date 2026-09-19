#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hw_bench.py — cross-platform hardware probe + micro-benchmark tool (stdlib only).

PURPOSE
    Probe CPU / memory / disk / GPU / software toolchain of a CI runner and run
    small, deterministic micro-benchmarks (single-core, multi-core, memory copy,
    sequential / random disk IO, optional network, optional native-tool
    cross-checks). Results are emitted as JSON (machine readable) and Markdown
    (human readable, appended into 07-bench.md).

SCORE FORMULA
    cpu score = 100 * geomean( sha256_mibps        / 1000,
                               sieve_primes_per_sec / 3.0e7,
                               f64_mflops           / 15 )
    The three denominators are the measured magnitude of a single core on a
    typical GitHub standard hosted runner (4 vCPU x86-64), so score ~= 100 means
    "on par with one core of a stock hosted runner", and the score is a pure
    relative index with no physical unit.
    The same formula is applied to cpu_multi using the aggregated (summed)
    worker throughputs. parallel_efficiency is defined as:
        parallel_efficiency = cpu_multi.score / (cpu_single.score * workers)
    so a perfect linear scaling yields ~1.0.

CROSS-PLATFORM COMPARABILITY NOTE
    Micro-benchmark numbers depend heavily on the CPython build and version.
    Numbers are only comparable across machines when the SAME Python version
    (ideally the same build) is used — the report must always record the
    Python version (meta.python) alongside the scores.

Runtime: Python 3.9+ (no match statements, no PEP-604 unions, no tomllib),
standard library only; numpy is used opportunistically when importable.
"""

import argparse
import ctypes
import datetime
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import random
import re
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.request

TOOL_NAME = "hw_bench"
TOOL_VERSION = "1.0.0"
MIB = 1024 * 1024
# 5s 太紧：github.com 首页有几百 KB，托管 runner 上一抖就整行网络指标全废。
# socket.create_connection 设的 timeout 会一直作用到后续 TLS 握手和 recv，所以给足余量。
NET_TIMEOUT = 20
SIEVE_LIMIT = 2_000_000  # fixed by contract
SIEVE_PRIMES_2M = 148933  # known prime count below 2e6 (result sanity check)
# sysbench cpu 的质数上界，必须显式传给 sysbench：各平台默认值不一致，
# 而这个参数直接决定 events/s 的量级（详见 bench_native 里的注释）
CPU_MAX_PRIME = 10000
# sysbench cpu 在 CPU_MAX_PRIME=10000 下的物理可行性上界。
#
# 为什么要这个守卫：真实 CI 上 macOS 报出 9,508,754 events/s，而 ubuntu 同参数、
# 同 sysbench 版本（1.0.20）、回显的 "Prime numbers limit" 也都是 10000 —— 差 2590 倍。
# 这不可能是硬件差异。反推：若事件数约 2.8 万（3 秒 × ~9500/s，与 Linux 实测同量级），
# 则 sysbench 认为耗时只有 0.003 秒，即它自己的计时错了三个数量级
# （macOS 上 clock_gettime 的单位处理问题，非本工具可控）。
#
# 上界的推导：prime 上界 10000 意味着每个 event 要对 10000 以内的候选数做试除，
# 即便按 1e9 次基本运算/秒算，单个 event 也不可能低于 ~10 微秒，
# 因此 events/s 的物理上限在 1e5 量级。实测真实值：EPYC runner 3,672、
# 本地 2 vCPU 容器 4,464。取 1e5 相对任何真实观测都有 >20x 余量，
# 同时能拦下 9.5e6 这类明显是计时错误的数字。
# 这不是"猜测合理性"：它只拒绝超出物理可能的值，被拒时原始行、版本、
# 回显参数全部保留在 JSON 里，读者可以自己复核。
SYSBENCH_CPU_MAX_PLAUSIBLE = 1.0e5


def _log(msg):
    print("[hw_bench] " + str(msg), flush=True)


def _read_text(path, limit=65536):
    try:
        with open(path, "r", errors="replace") as fh:
            return fh.read(limit)
    except OSError:
        return None


def _read_first_int(path):
    txt = _read_text(path)
    if txt is None:
        return None
    try:
        return int(txt.strip().split()[0])
    except (ValueError, IndexError):
        return None


def _cmd_output(cmd, timeout=8):
    """Run a command, return (rc, stdout_text, stderr_text); None on launch failure."""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=timeout
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
    err = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
    return proc.returncode, out, err


def _first_line(text, max_len=120):
    if not text:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:max_len]
    return None


def _run(name, fn):
    """Uniform wrapper: run one probe/benchmark, never raise, log one line."""
    t0 = time.perf_counter()
    try:
        result = fn()
        status = "ok"
    except Exception as exc:  # noqa: BLE001 - single-item failures are recorded
        result = {"error": "%s: %s" % (type(exc).__name__, exc)}
        status = "ERROR"
    _log("%-28s %-5s %7.2fs" % (name, status, time.perf_counter() - t0))
    return result


def _mibps(nbytes, seconds):
    if seconds <= 0:
        raise ValueError("non-positive elapsed time")
    return (nbytes / MIB) / seconds


def _score(sha256_mibps, sieve_primes_per_sec, f64_mflops):
    # 分母 = 典型 GitHub 托管 runner 单核实测量级，使 score≈100 表示与之一持平。
    # 详见模块 docstring 的 SCORE FORMULA 一节。
    ratios = []
    for value, denom in ((sha256_mibps, 1000.0),
                         (sieve_primes_per_sec, 3.0e7),
                         (f64_mflops, 15.0)):
        if value is None or value <= 0:
            raise ValueError("non-positive score component")
        ratios.append(value / denom)
    geo = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    return geo * 100.0


def _probe_cpu():
    system = platform.system()
    logical = os.cpu_count()
    physical = None
    model = None

    if system == "Linux":
        cpuinfo = _read_text("/proc/cpuinfo", 262144) or ""
        cores = set()
        phys_id = core_id = None
        for line in cpuinfo.splitlines():
            if line.startswith("physical id"):
                try:
                    phys_id = int(line.split(":", 1)[1])
                except ValueError:
                    phys_id = None
            elif line.startswith("core id"):
                try:
                    core_id = int(line.split(":", 1)[1])
                except ValueError:
                    core_id = None
            elif line.startswith("processor") and phys_id is not None:
                if core_id is not None:
                    cores.add((phys_id, core_id))
                phys_id = core_id = None
            elif not line.strip():
                if phys_id is not None and core_id is not None:
                    cores.add((phys_id, core_id))
                phys_id = core_id = None
        if cores:
            physical = len(cores)
        if not physical:
            # fall back to lscpu "Core(s) per socket" x "Socket(s)"
            res = _cmd_output(["lscpu"])
            if res:
                per_socket = sockets = None
                for line in res[1].splitlines():
                    if ":" not in line:
                        continue
                    key, val = line.split(":", 1)
                    if key.strip() == "Core(s) per socket":
                        per_socket = int(val.strip() or "0") if val.strip().isdigit() else None
                    elif key.strip() == "Socket(s)":
                        sockets = int(val.strip() or "0") if val.strip().isdigit() else None
                if per_socket and sockets:
                    physical = per_socket * sockets
        for line in cpuinfo.splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()[:120]
                break
        if not model:
            res = _cmd_output(["lscpu"])
            if res:
                for line in res[1].splitlines():
                    if line.startswith("Model name:"):
                        model = line.split(":", 1)[1].strip()[:120]
                        break
    elif system == "Darwin":
        res = _cmd_output(["sysctl", "-n", "hw.physicalcpu"])
        if res and res[0] == 0:
            try:
                physical = int(res[1].strip())
            except ValueError:
                physical = None
        res = _cmd_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        if res and res[0] == 0 and res[1].strip():
            model = res[1].strip()[:120]
        if not model and platform.machine() == "arm64":
            model = "Apple Silicon (arm64)"
    else:  # Windows and anything else
        model = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER") or None
        if model:
            model = model[:120]
        nproc = os.environ.get("NUMBER_OF_PROCESSORS")
        if nproc and nproc.isdigit() and not logical:
            logical = int(nproc)
    if not model:
        model = platform.processor() or None

    # cgroup CPU quota (Linux only) — reveals container runner limits (e.g. ubuntu-slim)
    quota_us = period_us = limit_cores = None
    if system == "Linux":
        txt = _read_text("/sys/fs/cgroup/cpu.max")  # cgroup v2
        if txt is not None:
            parts = txt.split()
            if len(parts) >= 2 and parts[0] != "max":
                try:
                    quota_us = int(parts[0])
                    period_us = int(parts[1])
                except ValueError:
                    quota_us = period_us = None
        else:  # cgroup v1
            q = _read_first_int("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
            p = _read_first_int("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
            if q is not None and p is not None:
                quota_us, period_us = q, p
        if quota_us and period_us and quota_us > 0 and period_us > 0:
            limit_cores = round(quota_us / period_us, 3)

    return {
        "logical_cores": logical,
        "physical_cores": physical,
        "arch": platform.machine(),
        "model": model,
        "cgroup_cpu_quota_us": quota_us,
        "cgroup_cpu_period_us": period_us,
        "cgroup_cpu_limit_cores": limit_cores,
    }


def _probe_memory():
    system = platform.system()
    if system == "Linux":
        info = {}
        meminfo = _read_text("/proc/meminfo") or ""
        for line in meminfo.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                parts = val.split()
                if parts and parts[0].isdigit():
                    info[key] = int(parts[0]) * 1024  # kB -> bytes
        total = info.get("MemTotal")
        avail = info.get("MemAvailable")
        if total is None:  # last resort
            avail = None
            total = None
        return {
            "total_mib": round(total / MIB, 1) if total else None,
            "available_mib": round(avail / MIB, 1) if avail else None,
            "swap_total_mib": round(info["SwapTotal"] / MIB, 1) if info.get("SwapTotal") else 0.0,
            "swap_free_mib": round(info["SwapFree"] / MIB, 1) if "SwapFree" in info else None,
        }
    if system == "Darwin":
        total = None
        res = _cmd_output(["sysctl", "-n", "hw.memsize"])
        if res and res[0] == 0:
            try:
                total = int(res[1].strip())
            except ValueError:
                total = None
        avail = None
        res = _cmd_output(["vm_stat"])
        if res and res[0] == 0:
            page = 4096
            head = res[1].splitlines()
            if head and head[0].startswith("Mach virtual memory statistics"):
                # header line ends with "page size of 4096 bytes"
                m = re.search(r"page size of (\d+) bytes", head[0])
                if m:
                    page = int(m.group(1))
            pages = 0
            for line in head[1:]:
                m = re.match(r"(Pages free|Pages inactive|Pages speculative):\s+(\d+)", line)
                if m:
                    pages += int(m.group(2))
            avail = pages * page
        swap_total = swap_free = None
        res = _cmd_output(["sysctl", "-n", "vm.swapusage"])
        if res and res[0] == 0:
            m = re.search(r"total = ([\d.]+)([GMK])", res[1])
            if m:
                mult = {"G": 1024.0, "M": 1.0, "K": 1.0 / 1024.0}
                swap_total = round(float(m.group(1)) * mult.get(m.group(2), 1.0), 1)
            m = re.search(r"free = ([\d.]+)([GMK])", res[1])
            if m:
                mult = {"G": 1024.0, "M": 1.0, "K": 1.0 / 1024.0}
                swap_free = round(float(m.group(1)) * mult.get(m.group(2), 1.0), 1)
        return {
            "total_mib": round(total / MIB, 1) if total else None,
            "available_mib": round(avail / MIB, 1) if avail else None,
            "swap_total_mib": swap_total,
            "swap_free_mib": swap_free,
        }
    # Windows (and fallback)
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return {
                "total_mib": round(stat.ullTotalPhys / MIB, 1),
                "available_mib": round(stat.ullAvailPhys / MIB, 1),
                "swap_total_mib": round(stat.ullTotalPageFile / MIB, 1),
                "swap_free_mib": round(stat.ullAvailPageFile / MIB, 1),
            }
    except (AttributeError, OSError):
        pass
    return {"total_mib": None, "available_mib": None,
            "swap_total_mib": None, "swap_free_mib": None}


def _mount_fs_type(path):
    """Best-effort filesystem type for the mount containing `path`."""
    system = platform.system()
    real = os.path.realpath(path)
    if system == "Linux":
        mounts = _read_text("/proc/self/mounts", 262144) or ""
        best_mp, best_fs = None, None
        for line in mounts.splitlines():
            fields = line.split()
            if len(fields) < 3:
                continue
            mp = fields[1].encode("latin-1", errors="replace").decode("unicode_escape")
            mp = mp.replace("\\040", " ").replace("\\011", "\t")
            if best_mp is not None and len(mp) <= len(best_mp):
                continue
            if real == mp or real.startswith(mp.rstrip("/") + "/") or real.startswith(mp + "/"):
                best_mp, best_fs = mp, fields[2]
        return best_fs
    if system == "Darwin":
        res = _cmd_output(["mount"])
        if res:
            best_mp, best_fs = None, None
            for line in res[1].splitlines():
                m = re.match(r"(.+?) on (.+?) \((.+)\)\s*$", line)
                if not m:
                    continue
                mp = m.group(2)
                if best_mp is not None and len(mp) <= len(best_mp):
                    continue
                if real == mp or real.startswith(mp.rstrip("/") + "/"):
                    best_mp = mp
                    best_fs = m.group(3).split(",")[0].strip()
            return best_fs
        return None
    if system == "Windows":
        try:
            drive = os.path.splitdrive(os.path.abspath(real))[0] + "\\"
            buf = ctypes.create_unicode_buffer(261)
            fs_buf = ctypes.create_unicode_buffer(261)
            ok = ctypes.windll.kernel32.GetVolumeInformationW(
                ctypes.c_wchar_p(drive), buf, 261, None, None, None, fs_buf, 261
            )
            if ok:
                return fs_buf.value or None
        except (AttributeError, OSError):
            pass
    return None


def _linux_mount_device(path):
    """从 /proc/mounts 找出 path 所属最长前缀挂载点的块设备，找不到返回 None。"""
    try:
        with open("/proc/mounts", encoding="utf-8", errors="replace") as fh:
            entries = [line.split() for line in fh if line.strip()]
    except OSError:
        return None
    ap = os.path.abspath(path)
    best = None
    for parts in entries:
        if len(parts) < 2:
            continue
        dev, mnt = parts[0], parts[1]
        mnt = mnt.replace("\\040", " ").replace("\\011", "\t")
        if mnt == "/" or ap == mnt or ap.startswith(mnt.rstrip("/") + "/"):
            if best is None or len(mnt) > len(best[0]):
                best = (mnt, dev)
    if not best:
        return None
    dev = best[1]
    # overlay / tmpfs / nfs 不是块设备，判断不了固态与否 —— 如实返回 None
    return dev if dev.startswith("/dev/") else None


def _disk_is_ssd(path):
    """尽力判断 path 落在 SSD 还是机械盘；判断不出来返回 None，**绝不猜**。"""
    system = platform.system()
    if system == "Linux":
        dev = _linux_mount_device(path)
        if not dev:
            return None
        base = os.path.basename(dev)
        # nvme0n1p2 -> nvme0n1 / mmcblk0p1 -> mmcblk0 / sda2 -> sda
        for pat in (r"^(nvme\d+n\d+)p\d+$", r"^(mmcblk\d+)p\d+$", r"^([a-z]+)\d+$"):
            m = re.match(pat, base)
            if m:
                base = m.group(1)
                break
        try:
            with open("/sys/block/%s/queue/rotational" % base, encoding="ascii") as fh:
                v = fh.read().strip()
        except OSError:
            return None
        return v == "0" if v in ("0", "1") else None
    if system == "Darwin":
        res = _cmd_output(["diskutil", "info", path], timeout=20)
        if res and res[0] == 0:
            m = re.search(r"Solid State:\s*(Yes|No)", res[1], re.I)
            if m:
                return m.group(1).lower() == "yes"
        return None
    if system == "Windows":
        # Get-PhysicalDisk 的 MediaType 才有 SSD/HDD/Unspecified 之分；
        # Win32_DiskDrive.MediaType 是固定字符串 "Fixed hard disk media"，没用
        script = ("try { $t = (Get-PhysicalDisk | Select-Object -First 1).MediaType; "
                  "if ($t) { Write-Output $t } } catch { }")
        res = _cmd_output(["powershell.exe", "-NoProfile", "-NonInteractive",
                           "-Command", script], timeout=40)
        if res and res[0] == 0:
            t = res[1].strip().lower()
            if t.startswith("ssd"):
                return True
            if t.startswith("hdd"):
                return False
        return None
    return None


def _probe_disk(disk_path):
    path = disk_path or tempfile.gettempdir()
    usage = shutil.disk_usage(path)
    return {
        "path": os.path.abspath(path),
        "total_mib": round(usage.total / MIB, 1),
        "free_mib": round(usage.free / MIB, 1),
        "used_percent": round(100.0 * usage.used / usage.total, 1) if usage.total else None,
        "filesystem": _mount_fs_type(path),
        "device": _linux_mount_device(path) if platform.system() == "Linux" else None,
        # True/False/None —— None 表示判断不出（overlay 挂载、容器 runner 等）
        "ssd": _disk_is_ssd(path),
    }


def _probe_gpu():
    # GitHub-hosted runners have NO discrete GPU — never fabricate one.
    res = _cmd_output(
        ["nvidia-smi", "--query-gpu=name,driver_version",
         "--format=csv,noheader"], timeout=8
    )
    if res and res[0] == 0 and res[1].strip():
        name = _first_line(res[1], 120)
        driver = None
        if name and "," in name:
            name, _, driver = [p.strip() for p in name.partition(",")]
            driver = driver[:120] or None
        return {"present": True, "name": name, "driver": driver,
                "note": "detected via nvidia-smi"}
    res = _cmd_output(["system_profiler", "SPDisplaysDataType"], timeout=15)
    if res and res[0] == 0 and "Chipset" in res[1]:
        line = _first_line(
            "\n".join(l for l in res[1].splitlines() if "Chipset" in l), 120
        )
        if line:
            return {"present": True, "name": line.split(":", 1)[-1].strip()[:120],
                    "driver": None, "note": "detected via system_profiler"}
    return {
        "present": False,
        "name": None,
        "driver": None,
        "note": "no discrete GPU detected (nvidia-smi / system_profiler unavailable or empty); GitHub-hosted runners normally have no GPU",
    }


def probe_hardware(disk_path):
    return {
        "cpu": _run("hardware.cpu", _probe_cpu),
        "memory": _run("hardware.memory", _probe_memory),
        "disk": _run("hardware.disk", lambda: _probe_disk(disk_path)),
        "gpu": _run("hardware.gpu", _probe_gpu),
    }


# (tool, version flag) — most use --version; java/ruby/xcodebuild use -version,
# openssl uses a bare "version" subcommand.
SOFTWARE_TOOLS = [
    ("pip", "--version"), ("node", "--version"), ("npm", "--version"),
    ("go", "version"), ("rustc", "--version"), ("cargo", "--version"),
    ("java", "-version"), ("javac", "-version"), ("dotnet", "--version"),
    ("gcc", "--version"), ("g++", "--version"), ("clang", "--version"),
    ("make", "--version"), ("cmake", "--version"), ("ninja", "--version"),
    ("git", "--version"), ("docker", "--version"), ("podman", "--version"),
    ("kubectl", "version"), ("helm", "version"), ("terraform", "version"),
    ("aws", "--version"), ("az", "--version"), ("gcloud", "--version"),
    ("jq", "--version"), ("yq", "--version"), ("curl", "--version"),
    ("wget", "--version"), ("7z", "--help"), ("7za", "--help"),
    ("unzip", "-v"), ("zip", "--version"), ("tar", "--version"),
    ("sqlite3", "--version"), ("mysql", "--version"), ("psql", "--version"),
    ("redis-cli", "--version"), ("mongo", "--version"), ("openssl", "version"),
    ("gpg", "--version"), ("pwsh", "--version"), ("ruby", "-version"),
    ("perl", "--version"), ("php", "--version"), ("swift", "--version"),
    ("xcodebuild", "-version"), ("choco", "--version"), ("brew", "--version"),
    ("apt", "--version"),
]


def _tool_version(name, flag):
    path = shutil.which(name)
    if not path:
        return None
    res = _cmd_output([path, flag], timeout=8)
    if res is None:
        return None
    rc, out, err = res
    line = _first_line(out) or _first_line(err)  # java -version prints to stderr
    if line is None:
        return None
    return line[:120]


def probe_software():
    tools = {}
    for name, flag in SOFTWARE_TOOLS:
        tools[name] = _tool_version(name, flag)
    # "python" records THIS interpreter (accurate and cheap, no subprocess)
    tools["python"] = platform.python_version()
    tools = {k: tools[k] for k in sorted(tools)}

    package_count = None
    site_dir = ""
    try:
        package_count = sum(1 for _ in importlib.metadata.distributions())
    except Exception:  # noqa: BLE001 - exotic installs may raise
        package_count = None
    try:
        import sysconfig
        site_dir = sysconfig.get_paths().get("purelib", "") or ""
    except Exception:  # noqa: BLE001
        site_dir = ""

    _log("software: %d tools probed, %d python packages" % (
        sum(1 for v in tools.values() if v), package_count or 0))
    return {"tools": tools, "python_packages": package_count,
            "python_site": site_dir}


# --------------------------------------------------------------------------
# CPU micro-benchmarks. Each takes its own `seconds` budget.
# --------------------------------------------------------------------------

def _bench_sha256(seconds):
    """SHA256 streaming throughput over a fixed 64 KiB buffer (MiB/s)."""
    buf = b"\xa5" * 65536
    h = hashlib.sha256()
    deadline = time.perf_counter() + seconds
    total = 0
    while True:
        for _ in range(64):  # amortize clock reads
            h.update(buf)
            total += 65536
        if time.perf_counter() >= deadline:
            break
    elapsed = time.perf_counter() - (deadline - seconds)
    if h.digest() == b"":  # keep the result alive; never true
        raise RuntimeError("unreachable")
    return _mibps(total, elapsed)


def _bench_int(seconds):
    """Pure-Python integer ALU throughput (ops/s)."""
    x = 0x123456789ABCDEF
    acc = 0
    deadline = time.perf_counter() + seconds
    iters = 0
    while True:
        for _ in range(20000):
            x = ((x ^ 0x9E3779B97F4A7C15) + (x << 3)) & 0xFFFFFFFFFFFFFFFF
            acc ^= x >> 17
        iters += 20000
        if time.perf_counter() >= deadline:
            break
    elapsed = time.perf_counter() - (deadline - seconds)
    if acc < 0:  # result must be consumed
        raise RuntimeError("unreachable")
    return iters * 3 / elapsed  # xor + shift-add + xor-shift ~= 3 ops


def _bench_f64(seconds):
    """FP64 throughput (MFLOPS) via sqrt / multiply-add loop."""
    x = 1.1
    acc = 0.0
    deadline = time.perf_counter() + seconds
    iters = 0
    while True:
        for _ in range(20000):
            x = math.sqrt(x * 1.0000001 + 1e-7)
            acc += x
        iters += 20000
        if time.perf_counter() >= deadline:
            break
    elapsed = time.perf_counter() - (deadline - seconds)
    if acc < 0.0:  # result must be consumed
        raise RuntimeError("unreachable")
    return iters * 3 / elapsed / 1e6  # mul + add + sqrt = 3 flops


def _sieve_once(limit):
    sieve = bytearray(b"\x01") * limit
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(range(i * i, limit, i)))
    return sieve.count(1)


def _bench_sieve(seconds, limit=SIEVE_LIMIT):
    """质数发现速率：每秒筛出的质数个数（= 筛法完成次数 × 每次的质数数）。

    注意别只返回 completions/elapsed —— 那是"每秒跑了几遍筛法"，量级差 SIEVE_PRIMES_2M
    倍，会让 score 被这一项拖垮且单位名不副实。
    """
    deadline = time.perf_counter() + seconds
    count = 0
    completions = 0
    while True:
        count = _sieve_once(limit)
        completions += 1
        if time.perf_counter() >= deadline:
            break
    elapsed = time.perf_counter() - (deadline - seconds)
    if limit == SIEVE_LIMIT and count != SIEVE_PRIMES_2M:
        raise RuntimeError("sieve sanity check failed: %d primes" % count)
    if count == -1:  # result must be consumed
        raise RuntimeError("unreachable")
    return completions * count / elapsed


def _cpu_loads(seconds):
    """Run the full single-core load set; returns raw metrics dict."""
    return {
        "sha256_mibps": _bench_sha256(seconds),
        "int_ops_per_sec": _bench_int(seconds),
        "f64_mflops": _bench_f64(seconds),
        "sieve_primes_per_sec": _bench_sieve(seconds),
    }


def bench_cpu_single(seconds):
    loads = _cpu_loads(seconds)
    result = dict(loads)
    result["sieve_limit"] = SIEVE_LIMIT
    result["score"] = _score(
        loads["sha256_mibps"], loads["sieve_primes_per_sec"], loads["f64_mflops"]
    )
    return result


# --------------------------------------------------------------------------
# Multi-core (spawn context; worker must stay a module-level function)
# --------------------------------------------------------------------------

def _mp_worker(seconds):
    """Picklable top-level worker: run the same load set, return metrics."""
    return _cpu_loads(seconds)


def bench_cpu_multi(seconds, workers, single_score):
    ctx = multiprocessing.get_context("spawn")
    if workers <= 0:
        workers = os.cpu_count() or 1
    with ctx.Pool(processes=workers) as pool:
        results = pool.map(_mp_worker, [seconds] * workers)
    agg = {k: sum(r[k] for r in results) for k in results[0]}
    score = _score(agg["sha256_mibps"], agg["sieve_primes_per_sec"],
                   agg["f64_mflops"])
    efficiency = score / (single_score * workers) if single_score else None
    return {
        "workers": workers,
        "sha256_mibps": agg["sha256_mibps"],
        "sieve_primes_per_sec": agg["sieve_primes_per_sec"],
        "score": score,
        "parallel_efficiency": efficiency,
    }


# --------------------------------------------------------------------------
# Memory copy bandwidth
# --------------------------------------------------------------------------

def bench_memory(seconds, mib=64):
    src = bytearray(b"\x5a" * (mib * MIB))
    dst = bytearray(len(src))
    deadline = time.perf_counter() + seconds
    copies = 0
    while True:
        dst[0:len(src)] = src
        copies += 1
        if time.perf_counter() >= deadline:
            break
    elapsed = time.perf_counter() - (deadline - seconds)
    if dst[:7] == b"nope":  # keep the destination alive
        raise RuntimeError("unreachable")
    copy_mibps = _mibps(copies * len(src), elapsed)

    numpy_mibps = None
    try:
        import numpy as np  # optional, stdlib-only contract preserved
        arr = np.frombuffer(bytes(src), dtype=np.uint8)
        # 必须复用同一块目标缓冲。arr.copy() 每轮都会新分配 64 MiB，测到的其实是
        # 缺页中断 + 内核清零的开销而不是 memcpy 带宽（实测会低一个数量级，
        # 甚至低于纯 bytearray 拷贝，纯属测量假象）。
        dst_np = np.empty_like(arr)
        deadline = time.perf_counter() + seconds
        ncopies = 0
        while True:
            np.copyto(dst_np, arr)
            ncopies += 1
            if time.perf_counter() >= deadline:
                break
        n_elapsed = time.perf_counter() - (deadline - seconds)
        if int(dst_np[0]) < 0:  # keep result alive
            raise RuntimeError("unreachable")
        numpy_mibps = _mibps(ncopies * arr.nbytes, n_elapsed)
    except Exception:
        # numpy 缺失或损坏都不能连累 bytearray 那一项的结果
        numpy_mibps = None

    return {"copy_mibps": copy_mibps, "mib": mib,
            "numpy_copy_mibps": numpy_mibps}


# --------------------------------------------------------------------------
# Disk benchmarks (temp dir under --disk-path, always cleaned in finally)
# --------------------------------------------------------------------------

def bench_disk_seq(disk_path, disk_mb, seconds):
    tmpdir = tempfile.mkdtemp(prefix="hw_bench_seq_", dir=disk_path)
    path = os.path.join(tmpdir, "seq.bin")
    try:
        buf = os.urandom(MIB)

        # 1) plain buffered write (no fsync)
        t0 = time.perf_counter()
        with open(path, "wb") as fh:
            for _ in range(disk_mb):
                fh.write(buf)
        write_mibps = _mibps(disk_mb * MIB, time.perf_counter() - t0)

        # 2) write + os.fsync (durable)
        t0 = time.perf_counter()
        with open(path, "wb") as fh:
            for _ in range(disk_mb):
                fh.write(buf)
            fh.flush()
            os.fsync(fh.fileno())
        fsync_mibps = _mibps(disk_mb * MIB, time.perf_counter() - t0)

        # 3) read back (page cache likely warm — we cannot drop caches unprivileged)
        t0 = time.perf_counter()
        total_read = 0
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(MIB)
                if not chunk:
                    break
                total_read += len(chunk)
        read_mibps = _mibps(total_read, time.perf_counter() - t0)
        if total_read != disk_mb * MIB:
            raise RuntimeError("short read: %d bytes" % total_read)

        return {
            "size_mb": disk_mb,
            "write_mibps": write_mibps,
            "read_mibps": read_mibps,
            "fsync_write_mibps": fsync_mibps,
            "page_cache_warm_read": True,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def bench_disk_rand(disk_path, disk_mb, seconds):
    block = 4096
    size_mb = min(max(disk_mb, 16), 256)
    nblocks = size_mb * MIB // block
    tmpdir = tempfile.mkdtemp(prefix="hw_bench_rand_", dir=disk_path)
    path = os.path.join(tmpdir, "rand.bin")
    try:
        with open(path, "wb") as fh:  # preallocate
            zero = b"\x00" * MIB
            for _ in range(size_mb):
                fh.write(zero)
        rnd = random.Random(0xC0FFEE)
        wbuf = bytes(rnd.getrandbits(8) for _ in range(256)) * 16  # 4 KiB

        # random 4 KiB writes: f.seek() + write (portable on Windows too)
        deadline = time.perf_counter() + seconds
        wops = 0
        with open(path, "r+b") as fh:
            while True:
                for _ in range(256):
                    fh.seek(rnd.randrange(nblocks) * block)
                    fh.write(wbuf)
                    wops += 1
                if time.perf_counter() >= deadline:
                    break
        w_elapsed = time.perf_counter() - (deadline - seconds)
        r4k_write_iops = wops / w_elapsed

        # random 4 KiB reads: f.seek() + read
        checksum = 0
        deadline = time.perf_counter() + seconds
        rops = 0
        with open(path, "rb") as fh:
            while True:
                for _ in range(256):
                    fh.seek(rnd.randrange(nblocks) * block)
                    data = fh.read(block)
                    checksum ^= len(data)
                    rops += 1
                if time.perf_counter() >= deadline:
                    break
        r_elapsed = time.perf_counter() - (deadline - seconds)
        if checksum < 0:  # result must be consumed
            raise RuntimeError("unreachable")
        r4k_read_iops = rops / r_elapsed

        return {
            "block_kb": block // 1024,
            "ops": wops + rops,
            "r4k_read_iops": r4k_read_iops,
            "r4k_write_iops": r4k_write_iops,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --------------------------------------------------------------------------
# Network (only with --net; every step has a 5 s timeout + try/except)
# --------------------------------------------------------------------------

def bench_net(url):
    out = {"url": url, "dns_ms": None, "tcp_connect_ms": None,
           "tls_handshake_ms": None, "ttfb_ms": None, "http_get_mibps": None,
           "bytes": None}
    errors = []
    host = port = path = None
    from urllib.parse import urlparse
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return dict(out, error="invalid URL: no hostname")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"

    # DNS
    try:
        t0 = time.perf_counter()
        socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        out["dns_ms"] = (time.perf_counter() - t0) * 1000.0
    except (OSError, socket.gaierror) as exc:
        errors.append("dns: %s" % exc)

    # raw TCP + TLS + TTFB on one connection
    try:
        t0 = time.perf_counter()
        sock = socket.create_connection((host, port), timeout=NET_TIMEOUT)
        out["tcp_connect_ms"] = (time.perf_counter() - t0) * 1000.0
        try:
            tls = None
            if parsed.scheme == "https":
                t0 = time.perf_counter()
                ctx = ssl.create_default_context()
                tls = ctx.wrap_socket(sock, server_hostname=host)
                out["tls_handshake_ms"] = (time.perf_counter() - t0) * 1000.0
                conn = tls
            else:
                conn = sock
            try:
                req = ("GET %s HTTP/1.1\r\nHost: %s\r\n"
                       "User-Agent: hw_bench/%s\r\nConnection: close\r\n\r\n"
                       % (path, host, TOOL_VERSION)).encode("ascii")
                conn.sendall(req)
                t0 = time.perf_counter()
                first = conn.recv(8192)
                out["ttfb_ms"] = (time.perf_counter() - t0) * 1000.0
                if not first:
                    raise OSError("empty response")
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
        finally:
            try:
                sock.close()
            except OSError:
                pass
    except (OSError, ssl.SSLError) as exc:
        errors.append("tcp/tls/ttfb: %s" % exc)

    # full HTTP GET throughput via urllib
    try:
        t0 = time.perf_counter()
        with urllib.request.urlopen(url, timeout=NET_TIMEOUT) as resp:
            body = resp.read()
        elapsed = time.perf_counter() - t0
        out["bytes"] = len(body)
        out["http_get_mibps"] = _mibps(len(body), elapsed)
    except Exception as exc:  # noqa: BLE001 - urllib raises many types
        errors.append("http_get: %s" % exc)

    if errors:
        # 只有全军覆没才标 error；部分成功时把失败原因降级成 warnings，
        # 否则汇总端会把已经测到的 dns_ms / ttfb_ms 一起判成失败，白丢数据
        got = any(out[k] is not None for k in
                  ("dns_ms", "tcp_connect_ms", "tls_handshake_ms", "ttfb_ms",
                   "http_get_mibps"))
        msg = "; ".join(errors)
        if got:
            out["warnings"] = msg
        else:
            out["error"] = msg
    return out


# --------------------------------------------------------------------------
# Native tool cross-checks (best effort; missing tools => null, not failure)
# --------------------------------------------------------------------------

def bench_native(seconds):
    result = {
        "sysbench_cpu_events_per_sec": None,
        "sysbench_memory_mibs": None,
        # 下面三个是审计用原始串：跨平台数字出现过 2200 倍的离谱差异，
        # 把参数和原始输出行一起存下来，才能核对而不是盲信一个浮点数
        "sysbench_cpu_args": None,
        "sysbench_cpu_raw": None,
        "sysbench_version": None,
        "sysbench_prime_limit_reported": None,
        "sysbench_cpu_note": None,
        "7z_mips": None,
        "7z_mips_compress": None,
        "7z_mips_decompress": None,
        "7z_threads": None,
        "7z_raw_tot": None,
        "7z_raw_avr": None,
    }
    t = max(1, int(seconds))
    if shutil.which("sysbench"):
        # sysbench 的语法是 `sysbench [options] [testname] [command]`，而且 --time 必须用
        # `--time=N` 形式。写成 `sysbench cpu --time 3 run`（选项在 testname 之后 + 空格分隔）
        # 会直接报 "Unrecognized command line argument: run"，实测三个平台全是这个死法，
        # 结果就是 native 段永远三个 None。
        #
        # --cpu-max-prime 必须显式钉住。实测它的默认值在不同发行版/构建之间不一致，
        # 而这个参数直接决定 events/s 的量级：
        #     max-prime=100   -> 2,474,937 events/s
        #     max-prime=1000  ->   118,950 events/s
        #     max-prime=10000 ->     4,434 events/s
        # 真实 CI 里 macOS 报出 8,150,296（Linux 同参数是 3,672），差 2200 倍，
        # 就是因为各平台默认值不同 —— 不钉参数的跨平台数字毫无可比性。
        cpu_args = ["sysbench", "--threads=1", "--time=%d" % t,
                    "--cpu-max-prime=%d" % CPU_MAX_PRIME, "cpu", "run"]
        result["sysbench_cpu_args"] = " ".join(cpu_args[1:])
        vres = _cmd_output(["sysbench", "--version"], timeout=10)
        if vres and vres[0] == 0:
            result["sysbench_version"] = _first_line(vres[1], 80)
        res = _cmd_output(cpu_args, timeout=t + 60)
        if res and res[0] == 0:
            out = res[1]
            m = re.search(r"^.*events per second:.*$", out, re.M)
            if m:
                # 存原始行，让任何人能核对这个数字是怎么来的，而不是只信一个浮点数
                result["sysbench_cpu_raw"] = m.group(0).strip()[:120]
            # sysbench 会把它实际采用的质数上界回显在 "Prime numbers limit: N"。
            # 显式传了 --cpu-max-prime 也照样可能被某些构建忽略：真实 CI 里 macOS 报出
            # 6,680,116 events/s，而参数串证明 --cpu-max-prime=10000 已经传进去了
            # （Linux 同参数是 3,659）。这个数量级只可能对应 prime≈20，说明该选项未生效。
            # 与其发布一个解释不了的数字，不如用回显值校验，对不上就拒绝采纳并记录原因。
            lim = re.search(r"Prime numbers limit:\s*(\d+)", out)
            result["sysbench_prime_limit_reported"] = (
                int(lim.group(1)) if lim else None)
            eps = re.search(r"events per second:\s*([\d.]+)", out)
            if eps:
                val = float(eps.group(1))
                lim_ok = lim is not None and int(lim.group(1)) == CPU_MAX_PRIME
                # 两道关卡都过才采纳。任一不过都把原始行留在 JSON 里，
                # 报告里标成"已拒采"而不是发布一个解释不了的数字
                if lim_ok and val <= SYSBENCH_CPU_MAX_PLAUSIBLE:
                    result["sysbench_cpu_events_per_sec"] = val
                else:
                    why = []
                    if not lim_ok:
                        why.append("请求 --cpu-max-prime=%d，但 sysbench 回显的是 %s"
                                   % (CPU_MAX_PRIME,
                                      lim.group(1) if lim else "未回显"))
                    if val > SYSBENCH_CPU_MAX_PLAUSIBLE:
                        why.append(
                            "events/s=%.0f 超出 prime 上界 %d 时的物理可行上限 %.0f "
                            "（真实硬件实测量级为 1e3~1e4），几乎可以确定是 sysbench "
                            "自身计时错误而非硬件极快" % (val, CPU_MAX_PRIME,
                                                          SYSBENCH_CPU_MAX_PLAUSIBLE))
                    result["sysbench_cpu_note"] = (
                        "已拒采：" + "；".join(why) +
                        "。原始输出行：%s；sysbench 版本：%s。"
                        "该平台的 sysbench CPU 指标不可与其他平台比较。"
                        % (result.get("sysbench_cpu_raw") or "n/a",
                           result.get("sysbench_version") or "n/a"))
        mem_args = ["sysbench", "--threads=1", "--time=%d" % t, "memory", "run"]
        res = _cmd_output(mem_args, timeout=t + 60)
        if res and res[0] == 0:
            # 形如 "6393.84 MiB transferred (6388.04 MiB/sec)"
            m = re.search(r"([\d.]+)\s*MiB/sec", res[1])
            if m:
                result["sysbench_memory_mibs"] = float(m.group(1))

    exe = None
    for cand in ("7z", "7za"):
        if shutil.which(cand):
            exe = cand
            break
    if exe is None and platform.system() == "Windows":
        # GitHub 的 Windows runner 预装了 7-Zip，但 C:\Program Files\7-Zip 不在 PATH 上
        # （仓库另一个 workflow pip-download-release.yml 用的也是这个绝对路径），
        # 所以 shutil.which 必然找不到，必须兜底扫绝对路径。
        for p in (r"C:\Program Files\7-Zip\7z.exe",
                  r"C:\Program Files (x86)\7-Zip\7z.exe"):
            if os.path.isfile(p):
                exe = p
                break
    if exe:
        res = _cmd_output([exe, "b"], timeout=900)
        if res and res[0] == 0:
            out = res[1]
            mt = re.search(r"#\s*Benchmark threads:\s*(\d+)", out)
            if mt:
                result["7z_threads"] = int(mt.group(1))
            # 7z b 的输出尾部形如（列：Speed KiB/s | Usage % | R/U MIPS | Rating MIPS，
            # "|" 前是压缩、后是解压）：
            #   Avr:      4921   100   5131   5120  |      84241   100   7340   7342
            #   Tot:             100   6235   6231
            # 原先的正则 r"Avr:\s*([\d.]+)\s*MIPS" 永远匹配不上 —— Avr 行里根本没有
            # "MIPS" 字面量，表头才有。Tot 行最后一个数才是常被引用的总分。
            # 这里不锚定行尾（$）：实测 macOS 上 7z 版本不同时列数/排版会变，
            # 锚定行尾会导致 Avr 整行匹配失败（真实 CI 里 macOS 的压缩/解压分项就是 None）。
            def _nums(line):
                return re.findall(r"[\d.]+", line.replace("|", " "))

            m = re.search(r"^Avr:.*$", out, re.M)
            if m:
                # 存下整行：macOS 上的 7z 构建排版与 Linux 不同，Avr 行解析不出 8 列，
                # 与其静默给 None，不如留原始行让人自己看
                result["7z_raw_avr"] = m.group(0).strip()[:160]
                nums = _nums(m.group(0))
                if len(nums) >= 8:
                    result["7z_mips_compress"] = float(nums[3])
                    result["7z_mips_decompress"] = float(nums[7])
            m = re.search(r"^Tot:.*$", out, re.M)
            if m:
                result["7z_raw_tot"] = m.group(0).strip()[:120]
                nums = _nums(m.group(0))
                if nums:
                    result["7z_mips"] = float(nums[-1])
    return result


# --------------------------------------------------------------------------
# Markdown rendering (07-bench.md fragment: H2 title, H3 sections, no H1)
# --------------------------------------------------------------------------

def _fmt_num(value, digits=2):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    try:
        if float(value) == int(float(value)) and abs(float(value)) < 1e15:
            return "{:,.0f}".format(float(value))
        return "{:,.{d}f}".format(float(value), d=digits)
    except (TypeError, ValueError):
        return str(value)


def _fmt_val(value, digits=2):
    if isinstance(value, dict):
        if "error" in value:
            msg = str(value["error"]).replace("|", "\\|")[:60]
            return "err: %s" % msg
        return "n/a"
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return _fmt_num(value, digits)
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text[:80]


def _table(rows):
    lines = ["| 指标 | 数值 | 单位 |", "|---|---|---|"]
    for label, value, unit in rows:
        lines.append("| %s | %s | %s |" % (label, _fmt_val(value), unit))
    return "\n".join(lines)


def render_markdown(report):
    b = report.get("benchmarks", {})
    cpu1 = b.get("cpu_single")
    cpuN = b.get("cpu_multi")
    mem = b.get("memory")
    dseq = b.get("disk_seq")
    drand = b.get("disk_rand")
    net = b.get("net")
    nat = b.get("native")

    parts = ["## ⚡ 基准测试结果", ""]
    parts.append("### 单核 CPU")
    parts.append(_table([
        ("SHA256 吞吐", cpu1 and cpu1.get("sha256_mibps"), "MiB/s"),
        ("整数吞吐", cpu1 and cpu1.get("int_ops_per_sec"), "ops/s"),
        ("FP64 吞吐", cpu1 and cpu1.get("f64_mflops"), "MFLOPS"),
        ("素数筛（limit=%s）" % _fmt_val(cpu1 and cpu1.get("sieve_limit"), 0),
         cpu1 and cpu1.get("sieve_primes_per_sec"), "primes/s"),
        ("单核综合分 score", cpu1 and cpu1.get("score"), "分"),
    ]))
    parts.append("")
    parts.append("### 多核 CPU")
    parts.append(_table([
        ("worker 进程数", cpuN and cpuN.get("workers"), "个"),
        ("SHA256 吞吐（聚合）", cpuN and cpuN.get("sha256_mibps"), "MiB/s"),
        ("素数筛（聚合）", cpuN and cpuN.get("sieve_primes_per_sec"), "primes/s"),
        ("多核综合分 score", cpuN and cpuN.get("score"), "分"),
        ("并行效率", cpuN and cpuN.get("parallel_efficiency"), ""),
    ]))
    parts.append("")
    parts.append("### 内存带宽")
    parts.append(_table([
        ("bytearray 拷贝带宽", mem and mem.get("copy_mibps"), "MiB/s"),
        ("拷贝块大小", mem and mem.get("mib"), "MiB"),
        ("numpy 拷贝带宽", mem and mem.get("numpy_copy_mibps"), "MiB/s"),
    ]))
    parts.append("")
    parts.append("### 磁盘顺序")
    parts.append(_table([
        ("写入吞吐", dseq and dseq.get("write_mibps"), "MiB/s"),
        ("写入吞吐（含 fsync）", dseq and dseq.get("fsync_write_mibps"), "MiB/s"),
        ("读回吞吐", dseq and dseq.get("read_mibps"), "MiB/s"),
        ("测试文件大小", dseq and dseq.get("size_mb"), "MB"),
        ("读测试命中页缓存", dseq and dseq.get("page_cache_warm_read"), ""),
    ]))
    if dseq and dseq.get("page_cache_warm_read"):
        parts.append("")
        parts.append("> 注：读回测试在写入后立即进行，可能命中页缓存"
                     "（非 root 环境无法 drop cache），数值偏乐观。")
    if platform.system() == "Darwin":
        parts.append("")
        parts.append("> ⚠️ **macOS 的 fsync 语义不同**：Darwin 上 `fsync()` 只把数据推到"
                     "驱动器缓存，并不强制落盘，真正的持久化刷盘需要 `fcntl(F_FULLFSYNC)`。"
                     "所以上表「写入吞吐（含 fsync）」在 macOS 上会显著高于 Linux / Windows"
                     "（实测约 20 倍），**不能跨平台直接比较**，只宜在 macOS 内部做纵向对比。")
    parts.append("")
    parts.append("### 磁盘随机 4K")
    parts.append(_table([
        ("随机读 IOPS", drand and drand.get("r4k_read_iops"), "IOPS"),
        ("随机写 IOPS", drand and drand.get("r4k_write_iops"), "IOPS"),
        ("块大小", drand and drand.get("block_kb"), "KiB"),
        ("总操作数", drand and drand.get("ops"), "次"),
    ]))
    parts.append("")
    parts.append("### 网络")
    if net is None:
        parts.append(_table([("未执行（未传 --net）", None, "")]))
    else:
        parts.append(_table([
            ("测试 URL", net.get("url"), ""),
            ("DNS 解析", net.get("dns_ms"), "ms"),
            ("TCP connect", net.get("tcp_connect_ms"), "ms"),
            ("TLS 握手", net.get("tls_handshake_ms"), "ms"),
            ("TTFB", net.get("ttfb_ms"), "ms"),
            ("HTTP GET 吞吐", net.get("http_get_mibps"), "MiB/s"),
            ("下载字节数", net.get("bytes"), "bytes"),
        ]))
        if net.get("error"):
            parts.append("")
            parts.append("> 注：网络步骤全部失败 — %s" %
                         _fmt_val(net.get("error")))
        elif net.get("warnings"):
            parts.append("")
            parts.append("> 注：部分网络步骤失败（已保留测到的数据）— %s" %
                         _fmt_val(net.get("warnings")))
    parts.append("")
    parts.append("### 原生工具交叉验证")
    parts.append(_table([
        ("sysbench cpu events/s（单线程）",
         nat and nat.get("sysbench_cpu_events_per_sec"), "events/s"),
        ("sysbench memory（单线程）", nat and nat.get("sysbench_memory_mibs"),
         "MiB/s"),
        ("7z b 总分（Tot Rating）", nat and nat.get("7z_mips"), "MIPS"),
        ("7z b 压缩 Rating（Avr）", nat and nat.get("7z_mips_compress"), "MIPS"),
        ("7z b 解压 Rating（Avr）", nat and nat.get("7z_mips_decompress"), "MIPS"),
        ("7z b 基准线程数", nat and nat.get("7z_threads"), "个"),
    ]))
    if nat is not None:
        # 把原始行贴出来：sysbench 的 events/s 曾在 macOS 上报出 8150296（Linux 是 3672），
        # 差 2200 倍。只给浮点数读者根本无从判断是硬件强还是参数不同，所以必须可审计。
        audit = []
        if nat.get("sysbench_version"):
            audit.append("- sysbench 版本：`%s`" % _fmt_val(nat.get("sysbench_version")))
        if nat.get("sysbench_cpu_args"):
            audit.append("- sysbench 实际参数：`%s`"
                         % _fmt_val(nat.get("sysbench_cpu_args")))
        if nat.get("sysbench_cpu_raw"):
            audit.append("- sysbench 原始输出行：`%s`"
                         % _fmt_val(nat.get("sysbench_cpu_raw")))
        if nat.get("sysbench_prime_limit_reported") is not None:
            audit.append("- sysbench 回显的质数上界：`%s`（请求 `%d`）"
                         % (_fmt_val(nat.get("sysbench_prime_limit_reported")),
                            CPU_MAX_PRIME))
        if nat.get("7z_raw_avr"):
            audit.append("- 7z 原始 Avr 行：`%s`" % _fmt_val(nat.get("7z_raw_avr")))
        if nat.get("7z_raw_tot"):
            audit.append("- 7z 原始 Tot 行：`%s`" % _fmt_val(nat.get("7z_raw_tot")))
        if audit:
            parts.append("")
            parts.append("**原始输出（可审计）**")
            parts.extend(audit)
        if nat.get("sysbench_cpu_note"):
            parts.append("")
            parts.append("> ⚠️ %s" % _fmt_val(nat.get("sysbench_cpu_note")))
    if nat is None:
        parts.append("")
        parts.append("> 注：未执行（传了 --skip-native）。")
    else:
        parts.append("")
        parts.append("> 注：`sysbench` 用 `--threads=1`，与上面的单核基准对齐；"
                     "`7z b` 默认用满全部线程，是多核负载，应与「多核 CPU」一节对照。"
                     "工具未安装时对应项为 n/a（Windows 上 7-Zip 装在 "
                     "`C:\\Program Files\\7-Zip`，不在 PATH，已按绝对路径兜底查找）。")
    parts.append("")
    return "\n".join(parts)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser():
    parser = argparse.ArgumentParser(
        prog="hw_bench.py",
        description="Hardware probe + micro-benchmark tool (stdlib only, Py3.9+)",
    )
    parser.add_argument("--seconds", type=float, default=3.0,
                        help="per-benchmark time budget in seconds (default 3)")
    parser.add_argument("--disk-mb", type=int, default=1024,
                        help="sequential disk test size in MB (default 1024)")
    parser.add_argument("--disk-path", default=tempfile.gettempdir(),
                        help="directory for disk benchmarks (default: temp dir)")
    parser.add_argument("--workers", type=int, default=0,
                        help="multi-core worker processes, 0 = all logical cores")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="write full results JSON to this path")
    parser.add_argument("--md", dest="md_path", default=None,
                        help="write markdown fragment (07-bench.md) to this path")
    parser.add_argument("--net", action="store_true",
                        help="run network benchmarks (default URL https://github.com)")
    parser.add_argument("--net-url", default="https://github.com",
                        help="URL for network benchmarks")
    parser.add_argument("--skip-native", action="store_true",
                        help="skip sysbench / 7z cross-validation")
    parser.add_argument("--quick", action="store_true",
                        help="fast mode: --seconds 1 --disk-mb 256")
    return parser


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
    args = build_arg_parser().parse_args(argv)
    seconds = 1.0 if args.quick else args.seconds
    disk_mb = 256 if args.quick else args.disk_mb
    disk_path = args.disk_path or tempfile.gettempdir()

    t_start = time.perf_counter()
    report = {}

    report["meta"] = {
        "tool": TOOL_NAME,
        "version": TOOL_VERSION,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "arch": platform.machine(),
        "runner_os": os.environ.get("RUNNER_OS") or platform.system(),
        "runner_arch": os.environ.get("RUNNER_ARCH"),
        "elapsed_s": None,  # filled below
        "argv": sys.argv[1:],
    }

    _log("hardware + software probing")
    report["hardware"] = probe_hardware(disk_path)
    report["software"] = _run("software", probe_software)

    benchmarks = {}
    cpu1 = _run("benchmarks.cpu_single", lambda: bench_cpu_single(seconds))
    benchmarks["cpu_single"] = cpu1
    single_score = cpu1.get("score") if isinstance(cpu1, dict) else None
    benchmarks["cpu_multi"] = _run(
        "benchmarks.cpu_multi",
        lambda: bench_cpu_multi(seconds, args.workers, single_score),
    )
    benchmarks["memory"] = _run("benchmarks.memory",
                                lambda: bench_memory(seconds))
    benchmarks["disk_seq"] = _run("benchmarks.disk_seq",
                                  lambda: bench_disk_seq(disk_path, disk_mb, seconds))
    benchmarks["disk_rand"] = _run("benchmarks.disk_rand",
                                   lambda: bench_disk_rand(disk_path, disk_mb, seconds))
    if args.net:
        benchmarks["net"] = _run("benchmarks.net",
                                 lambda: bench_net(args.net_url))
    else:
        benchmarks["net"] = None
    if not args.skip_native:
        benchmarks["native"] = _run("benchmarks.native",
                                    lambda: bench_native(seconds))
    else:
        benchmarks["native"] = None
    report["benchmarks"] = benchmarks

    report["meta"]["elapsed_s"] = round(time.perf_counter() - t_start, 2)

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        _log("json written: %s" % args.json_path)
    if args.md_path:
        with open(args.md_path, "w", encoding="utf-8") as fh:
            fh.write(render_markdown(report))
        _log("markdown written: %s" % args.md_path)

    _log("all done in %.1fs (exit 0)" % report["meta"]["elapsed_s"])
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
