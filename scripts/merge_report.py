#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""merge_report.py — 合并各平台 probe artifact 中的 bench.json，生成跨平台对比报告。

用法:
    python3 merge_report.py --root reports/ --out summary.md [--github-summary]

输入发现（递归 glob，两种布局都支持，不因目录层级假设而崩溃）:
  - 子目录模式（download-artifact 默认）: <root>/probe-<os>/bench.json
  - 平铺文件模式（merge-multiple 等）:     <root>/bench-<os>.json
  - 其他任意嵌套: 取 bench.json 的父目录名（去掉 probe- 前缀）作为平台名

输出:
  - --out 指定的 summary.md（H2 起，可直接贴进 $GITHUB_STEP_SUMMARY）
  - --out 同目录下的 merged.json（各平台原始数据 + 派生指标，供脚本化消费）

健壮性契约:
  平台 artifact 缺失 / bench.json 不存在 / JSON 非法 / 某项基准为 {"error": ...}
  / 某个值为 null —— 全部优雅降级为 `❌ 解析失败` / `❌ 错误: <截断60字>` /
  `n/a` / `未测试` / `-`，脚本退出码保持 0。
  仅当整个 --root 下找不到任何 bench.json 时，退出码 1 并向 stderr 打印明确错误。

兼容 Python 3.9+，仅标准库。
"""

import argparse
import fnmatch
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 平台排序优先级（fnmatch 通配），未命中者按字典序追加在最后
PLATFORM_PRIORITY = [
    "ubuntu-latest",
    "ubuntu-*-arm",
    "windows-latest",
    "windows-*-arm",
    "macos-latest",
    "macos-1*",
    "macos-2*",
    "ubuntu-slim",
]

ARCH_ALIASES = {
    "x86_64": "x64",
    "amd64": "x64",
    "x64": "x64",
    "em64t": "x64",
    "arm64": "arm64",
    "aarch64": "arm64",
    "arm": "arm",
    "i386": "x86",
    "i686": "x86",
}

# 可比性说明（固定段落）
COMPARABILITY_NOTES = """\
- GitHub 托管 runner 是**共享物理机上的 VM**，邻居负载可能造成同一标签两次运行最高 **±30%** 的波动；单次结果只能用于定性比较，重要结论请多次运行取中位数。
- **公开仓库的标准 runner 分钟数免费无限量**；私有仓库按分钟计费，且 **macOS 费率是 Linux 的 10 倍**（Windows 为 2 倍），跨平台跑全量矩阵前先算成本。
- `macos-latest` 现为 **M1 arm64、3 vCPU、7 GB RAM**（不是 4 核 16 GB），与 Intel mac runner（`macos-15-intel` / `macos-26-intel`，4 vCPU / 14 GB）**不可直接比较**。
- `ubuntu-slim` 是 **1 vCPU 的容器**（非 VM）、5 GB RAM、单 job 上限 **15 分钟**、**不支持 Docker-in-Docker**，多核基准可能因 cgroup 配额而失败或失真。
- GitHub 标准 runner **不含 GPU**；表中 GPU 行应如实显示"无"。
- 磁盘顺序读可能命中**页缓存**（runner 上无法 drop cache），`fsync` 写才更接近真实落盘性能；随机 4K IOPS 同理仅供参考。
- 纯 Python 基准受**解释器版本**影响显著，跨平台对比前必须核对 `Python 版本` 行是否一致；原生工具（sysbench / 7z）结果可作为交叉验证。\
"""

# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def trunc(s, n):
    """把字符串截断到 n 个字符（用省略号收尾），并压成单行。"""
    s = str(s).replace("\n", " ").replace("\r", " ").strip()
    return s if len(s) <= n else s[: max(1, n - 1)] + "…"


def esc(s):
    """Markdown 表格单元格转义：不能出现未转义 | 和换行。"""
    return str(s).replace("|", "/").replace("\n", " ").strip()


def fmt_num(v):
    """数字统一格式化：整数加千分位，浮点保留 1~2 位小数。"""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return "{:,}".format(v)
    v = float(v)
    if v == int(v) and abs(v) < 1e15:
        return "{:,}".format(int(v))
    if abs(v) >= 100:
        return "{:,.1f}".format(v)
    if abs(v) >= 10:
        return "{:,.2f}".format(v)
    return "{:,.3f}".format(v)


def fmt_pct(v):
    """0~1 的比率格式化为百分比。"""
    try:
        return "{:.0f}%".format(float(v) * 100)
    except (TypeError, ValueError):
        return "n/a"


def norm_arch(s):
    return ARCH_ALIASES.get(str(s).strip().lower(), str(s).strip())


def to_gb(v):
    """内存/磁盘容量归一到 GB：数值 > 100000 视为字节数（>97 GB 的裸数字不现实）。"""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    if v > 100000:
        return v / (1024.0 ** 3)
    return v


def pick(d, *names):
    """从 dict 里按候选键名取第一个非 None 值；再做一次大小写不敏感匹配。"""
    if not isinstance(d, dict):
        return None
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    low = {}
    for k, v in d.items():
        if isinstance(k, str) and v is not None:
            low[k.lower()] = v
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def dig(rec, *keys):
    """沿嵌套 dict 下钻，任何一层不是 dict 就返回 {}。"""
    d = rec.get("data")
    if not isinstance(d, dict):
        return {}
    for k in keys:
        v = d.get(k)
        if not isinstance(v, dict):
            return {}
        d = v
    return d


# ---------------------------------------------------------------------------
# 平台发现与加载
# ---------------------------------------------------------------------------


def platform_sort_key(name):
    """按固定优先级排序；未知标签安全地按字典序追加，绝不抛异常。"""
    try:
        for i, pat in enumerate(PLATFORM_PRIORITY):
            if fnmatch.fnmatchcase(name, pat):
                return (0, i, name)
        return (1, 0, name)
    except Exception:
        return (2, 0, str(name))


def platform_from_path(root, p):
    """从 bench.json 路径反推平台名。

    - 父目录名以 probe- 开头 -> 去前缀（download-artifact 子目录模式）
    - 文件直接在 root 下且名为 bench-<x>.json -> 平铺文件模式
    - 其他 -> 用父目录名（去 probe- 前缀）兜底
    """
    try:
        parent = p.parent
        if parent == root:
            m = re.match(r"^bench-(.+)\.json$", p.name, re.IGNORECASE)
            if m:
                return m.group(1)
            return "unknown"
        name = parent.name
        if name.lower().startswith("probe-"):
            name = name[len("probe-"):]
        return name or "unknown"
    except Exception:
        return "unknown"


def discover(root):
    """递归找出 root 下所有 bench.json / bench-<platform>.json，返回 {平台: Path}。"""
    root = Path(root)
    found = {}
    paths = []
    try:
        paths = sorted(set(root.rglob("bench.json")) | set(root.rglob("bench-*.json")))
    except Exception as e:  # 权限等异常不应让脚本崩
        print("WARN: 扫描 {} 失败: {}".format(root, e), file=sys.stderr)
        return found
    for p in paths:
        if not p.is_file():
            continue
        name = platform_from_path(root, p)
        if name == "unknown":
            print(
                "WARN: 无法从路径反推平台名: {}（将记为 unknown）".format(p),
                file=sys.stderr,
            )
        if name in found:
            print(
                "WARN: 平台 {} 出现多次，保留首个 {}，忽略 {}".format(
                    name, found[name], p
                ),
                file=sys.stderr,
            )
            continue
        found[name] = p

    # 「有 report.md 但没有 bench.json」的 artifact 也要收进来：
    # 只扫 bench.json 会让跑崩的平台从对比表里彻底消失，等于隐藏了最需要关注的失败
    known_dirs = {Path(v).parent for v in found.values()}
    for rp in sorted(root.rglob("report.md")):
        if rp.parent in known_dirs:
            continue
        name = platform_from_path(root, rp)
        if name in found:
            continue
        print(
            "WARN: 平台 {} 只有 report.md 没有 bench.json，记为失败平台".format(name),
            file=sys.stderr,
        )
        found[name] = rp
    return found


def load_platform(name, path, root):
    """读取单个平台的 bench.json；解析失败不抛异常，记录 error。"""
    rec = {
        "name": name,
        "source": str(path),
        "data": None,
        "error": None,
        "report_md_bytes": None,
    }
    p = Path(path)
    if p.suffix != ".json":
        # discover() 也会把「只有 report.md、没有 bench.json」的 artifact 收进来，
        # 否则该平台会静默消失，看不出 probe job 在基准阶段前就崩了
        rec["error"] = "未产出 bench.json（只有 {}，说明该平台 probe job 在基准阶段前就失败了）".format(p.name)
    else:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            data = json.loads(text)
            if isinstance(data, dict):
                rec["data"] = data
            else:
                rec["error"] = "bench.json 顶层不是 JSON 对象（实际为 {}）".format(
                    type(data).__name__
                )
        except Exception as e:
            rec["error"] = "{}: {}".format(type(e).__name__, e)
    # report.md 字节数（供第 5 节引用），找不到就保持 None
    candidates = [
        Path(path).parent / "report.md",
        Path(root) / "probe-{}".format(name) / "report.md",
        Path(root) / name / "report.md",
        Path(root) / "report-{}.md".format(name),
    ]
    for c in candidates:
        try:
            if c.is_file():
                rec["report_md_bytes"] = c.stat().st_size
                break
        except OSError:
            pass
    return rec


# ---------------------------------------------------------------------------
# 单元格取值：统一状态机
#   ("num", v)    数值，可参与冠军评选
#   ("text", s)   文本，不参与冠军评选
#   ("na",)       缺失 / null / 类型不对 -> "n/a"（或行自定义占位）
#   ("notest",)   net 专属：未启用网络基准 -> "未测试"
#   ("err", msg)  该基准段整体失败 -> "❌ 错误: <msg>"
#   ("dead",)     该平台 bench.json 解析失败 -> "❌ 解析失败"
# ---------------------------------------------------------------------------


def bench_state(rec, section, key):
    data = rec.get("data")
    if data is None:
        return ("dead",)
    b = data.get("benchmarks")
    if not isinstance(b, dict):
        b = {}
    sec = b.get(section)
    if sec is None:
        return ("notest",) if section == "net" else ("na",)
    if not isinstance(sec, dict):
        return ("na",)
    v = sec.get(key)
    if not isinstance(v, bool) and isinstance(v, (int, float)):
        return ("num", v)
    # 优先认这个指标自己的值；只有它确实没测到，才回落到整段级的 error。
    # hw_bench 在网络部分成功时只写 warnings 不写 error，所以部分数据能保住。
    if "error" in sec:
        return ("err", str(sec.get("error") or "未知错误"))
    return ("notest",) if section == "net" else ("na",)


def hw_dead(rec):
    if rec.get("data") is None:
        return ("dead",)
    return None


def render_cells(platforms, states, champ=None, fmt=fmt_num, na_text="n/a"):
    """把状态列表渲染成 Markdown 单元格；champ='max'/'min' 时给冠军值加粗。"""
    cells = []
    nums = {}
    for rec, st in zip(platforms, states):
        kind = st[0]
        if kind == "num":
            try:
                txt = fmt(st[1])
                nums[len(cells)] = float(st[1])
            except Exception:
                txt = "n/a"
        elif kind == "text":
            txt = esc(trunc(st[1], 60))
        elif kind == "err":
            txt = "❌ 错误: " + esc(trunc(st[1], 60))
        elif kind == "notest":
            txt = "未测试"
        elif kind == "dead":
            txt = "❌ 解析失败"
        else:
            txt = na_text
        cells.append(txt)
    if champ in ("max", "min") and nums:
        try:
            if champ == "max":
                best = max(nums.items(), key=lambda kv: kv[1])[0]
            else:
                best = min(nums.items(), key=lambda kv: kv[1])[0]
            cells[best] = "**{}**".format(cells[best])
        except Exception:
            pass
    return cells


def render_table(header_label, platforms, rows):
    """rows: [(行标签, cells 列表)]，列数 = 平台数，动态生成。"""
    ncol = len(platforms) + 1
    lines = []
    lines.append("| {} | {} |".format(header_label, " | ".join(
        esc(p["name"]) for p in platforms)))
    lines.append("|" + "---|" * ncol)
    for label, cells in rows:
        # 防御：列数不齐时补齐/截断，避免表格错位
        cells = list(cells)[: len(platforms)]
        while len(cells) < len(platforms):
            cells.append("n/a")
        lines.append("| {} | {} |".format(esc(label), " | ".join(cells)))
    return "\n".join(lines)


def error_notes(platforms):
    """解析失败 / 单项失败的注释行（放在表格外，避免单元格塞不下完整错误）。"""
    notes = []
    for rec in platforms:
        if rec.get("error"):
            notes.append(
                "> ❌ **{}**: bench.json 读取失败 — 错误: {}".format(
                    esc(rec["name"]), esc(trunc(rec["error"], 60))
                )
            )
            continue
        data = rec.get("data")
        if not isinstance(data, dict):
            continue
        b = data.get("benchmarks")
        if not isinstance(b, dict):
            continue
        # 段级失败/降级必须显式冒出来，否则「表里一片 n/a」和「真没测」无法区分
        for sec_name, sec in sorted(b.items()):
            if not isinstance(sec, dict):
                continue
            if sec.get("error"):
                notes.append(
                    "> ❌ **{}** / {}: {}".format(
                        esc(rec["name"]), esc(sec_name),
                        esc(trunc(str(sec["error"]), 120))
                    )
                )
            elif sec.get("warnings"):
                notes.append(
                    "> ⚠️ **{}** / {}: 部分指标失败 — {}".format(
                        esc(rec["name"]), esc(sec_name),
                        esc(trunc(str(sec["warnings"]), 120))
                    )
                )
    return notes


# ---------------------------------------------------------------------------
# 硬件 / 环境指标提取（键名兼容多种拼写——hw_bench.py 的子键未被契约钉死）
# ---------------------------------------------------------------------------


def x_arch(rec):
    st = hw_dead(rec)
    if st:
        return st
    cpu = dig(rec, "hardware", "cpu")
    meta = dig(rec, "meta")
    # meta.runner_arch 是 runner 自己上报的 X64/ARM64，最权威；
    # 本地跑（非 Actions 环境）时它是 null，再退回 cpu.arch / meta.machine
    a = pick(meta, "runner_arch") or pick(cpu, "arch", "architecture") or pick(
        meta, "arch", "machine"
    )
    if a is None:
        return ("na",)
    return ("text", norm_arch(a))


def x_cores(rec, *names):
    st = hw_dead(rec)
    if st:
        return st
    cpu = dig(rec, "hardware", "cpu")
    v = pick(cpu, *names)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
        return ("na",)
    return ("num", int(v))


def x_cpu_model(rec):
    st = hw_dead(rec)
    if st:
        return st
    cpu = dig(rec, "hardware", "cpu")
    v = pick(cpu, "model", "model_name", "brand", "brand_string", "cpu_model")
    if v is None:
        return ("na",)
    return ("text", trunc(v, 40))


def x_cgroup_cores(rec):
    st = hw_dead(rec)
    if st:
        return st
    cpu = dig(rec, "hardware", "cpu")
    v = pick(
        cpu,
        # hw_bench.py 写的就是这个键；其余是历史/别实现的兼容名
        "cgroup_cpu_limit_cores",
        "cgroup_effective_cores",
        "cgroup_equivalent_cores",
        "cgroup_cores",
        "cgroup_quota_cores",
        "effective_cores",
    )
    if v is None:
        return ("text", "-")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return ("text", "-")
    return ("text", fmt_num(v))


# 单位后缀 -> 换算到 GB 的除数。hw_bench.py 写的是 *_mib，早期实现/fixture 可能是
# *_bytes 或裸 *_gb，所以按后缀显式换算，别再用 to_gb 里「数量级 >100000 就当字节」
# 那种启发式——小容量盘（如 ubuntu-slim 的 14GB）会被误判。
_CAPACITY_UNITS = (
    ("bytes", 1024.0 ** 3), ("b", 1024.0 ** 3),
    ("kib", 1024.0 ** 2), ("kb", 1024.0 ** 2),
    ("mib", 1024.0), ("mb", 1024.0),
    ("gib", 1.0), ("gb", 1.0),
)

# (family, base) -> 候选键名。hw_bench 的 disk 用 free_mib、memory 用 swap_total_mib，
# 而对比表按语义叫 available / swap，需要别名映射。
_CAPACITY_ALIASES = {
    ("memory", "total"): ("total", "mem_total", "total_memory"),
    ("memory", "available"): ("available", "free", "mem_available"),
    ("memory", "swap"): ("swap_total", "swap", "swap_size"),
    ("disk", "total"): ("total", "size", "capacity"),
    ("disk", "available"): ("free", "available", "free_space", "avail"),
}


def x_capacity(rec, family, base):
    """容量类指标统一换算到 GB，单位感知。"""
    st = hw_dead(rec)
    if st:
        return st
    d = dig(rec, "hardware", family)
    if not isinstance(d, dict):
        return ("na",)
    low = {k.lower(): v for k, v in d.items() if isinstance(k, str)}
    aliases = _CAPACITY_ALIASES.get((family, base), (base,))
    for name in aliases:
        for unit, div in _CAPACITY_UNITS:
            for cand in (name + "_" + unit, name + unit):
                v = low.get(cand)
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    continue
                return ("num", float(v) / div)
        v = low.get(name)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        # 裸数字：按 to_gb 的旧启发式兜底，兼容第三方产的 bench.json
        gb = to_gb(v)
        if gb is not None:
            return ("num", gb)
    return ("na",)


def x_disk_fs(rec):
    st = hw_dead(rec)
    if st:
        return st
    d = dig(rec, "hardware", "disk")
    v = pick(d, "filesystem", "fs_type", "fstype", "fs")
    if v is None:
        return ("na",)
    return ("text", trunc(v, 30))


def x_disk_ssd(rec):
    st = hw_dead(rec)
    if st:
        return st
    d = dig(rec, "hardware", "disk")
    v = pick(d, "ssd", "is_ssd", "rotational")
    if v is None:
        return ("na",)
    if isinstance(v, bool):
        return ("text", "是" if v else "否")
    if isinstance(v, (int, float)):  # rotational: 0=SSD
        return ("text", "是" if v == 0 else "否")
    return ("na",)


def x_gpu(rec):
    st = hw_dead(rec)
    if st:
        return st
    data = rec.get("data") or {}
    hw = data.get("hardware")
    if not isinstance(hw, dict) or "gpu" not in hw:
        return ("na",)  # 整键缺失
    gpu = hw.get("gpu")
    if not isinstance(gpu, dict):
        return ("na",)
    if "present" not in gpu:
        return ("na",)
    return ("text", "有" if gpu.get("present") else "无")


def x_tool_version(rec, tool):
    st = hw_dead(rec)
    if st:
        return st
    meta = dig(rec, "meta")
    sw = dig(rec, "software")
    tools = pick(sw, "tools")
    if not isinstance(tools, dict):
        tools = {}
    v = pick(meta, tool) if tool == "python" else None
    v = v or pick(tools, tool)
    if v is None:
        return ("na",)
    return ("text", trunc(v, 40))


def x_python_packages(rec):
    st = hw_dead(rec)
    if st:
        return st
    sw = dig(rec, "software")
    v = pick(sw, "python_packages", "installed_packages", "packages")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return ("na",)
    return ("num", int(v))


def x_runner_image(rec):
    st = hw_dead(rec)
    if st:
        return st
    meta = dig(rec, "meta")
    sw = dig(rec, "software")
    parts = []
    ros = pick(meta, "runner_os") or pick(meta, "os")
    if ros:
        parts.append(trunc(ros, 20))
    img = pick(meta, "image_os", "image", "image_version") or pick(
        sw, "image_os", "image", "image_version", "runner_image", "runner_image_version"
    )
    if img:
        parts.append(trunc(img, 40))
    if not parts:
        return ("na",)
    return ("text", " / ".join(parts))


def build_hw_table(platforms):
    """第 1 节对比表。返回 (table_md, champions_dict)。"""
    specs = [
        ("架构", lambda r: x_arch(r), None, fmt_num),
        ("逻辑核数", lambda r: x_cores(r, "logical_cores", "logical", "cores_logical", "vcpus", "cpu_count"), "max", fmt_num),
        ("物理核数", lambda r: x_cores(r, "physical_cores", "physical", "cores_physical"), "max", fmt_num),
        ("CPU 型号", x_cpu_model, None, fmt_num),
        ("cgroup 等效核数", x_cgroup_cores, None, fmt_num),
        ("总内存 (GB)", lambda r: x_capacity(r, "memory", "total"), "max", fmt_num),
        ("可用内存 (GB)", lambda r: x_capacity(r, "memory", "available"), "max", fmt_num),
        ("交换分区 (GB)", lambda r: x_capacity(r, "memory", "swap"), "max", fmt_num),
        ("磁盘总容量 (GB)", lambda r: x_capacity(r, "disk", "total"), "max", fmt_num),
        ("磁盘可用 (GB)", lambda r: x_capacity(r, "disk", "available"), "max", fmt_num),
        ("文件系统", x_disk_fs, None, fmt_num),
        ("SSD", x_disk_ssd, None, fmt_num),
        ("GPU", x_gpu, None, fmt_num),
        ("Python 版本", lambda r: x_tool_version(r, "python"), None, fmt_num),
        ("Node 版本", lambda r: x_tool_version(r, "node"), None, fmt_num),
        ("GCC 版本", lambda r: x_tool_version(r, "gcc"), None, fmt_num),
        ("已装 Python 包数", x_python_packages, "max", fmt_num),
        ("Runner 镜像", x_runner_image, None, fmt_num),
    ]
    rows = []
    champions = {}
    for label, fn, champ, fmt in specs:
        states = [fn(rec) for rec in platforms]
        cells = render_cells(platforms, states, champ=champ, fmt=fmt)
        if champ and any(c.startswith("**") for c in cells):
            for rec, c in zip(platforms, cells):
                if c.startswith("**"):
                    champions[label] = rec["name"]
        rows.append((label, cells))
    return render_table("指标", platforms, rows), champions


# ---------------------------------------------------------------------------
# 第 2 节：基准成绩对比
# ---------------------------------------------------------------------------

BENCH_ROWS = [
    # (行标签, section, key, champion, fmt, na_text)
    ("SHA256 单核 (MiB/s)", "cpu_single", "sha256_mibps", "max", fmt_num, "n/a"),
    ("素数筛单核 (primes/s)", "cpu_single", "sieve_primes_per_sec", "max", fmt_num, "n/a"),
    ("INT 单核 (ops/s)", "cpu_single", "int_ops_per_sec", "max", fmt_num, "n/a"),
    ("FP64 单核 (MFLOPS)", "cpu_single", "f64_mflops", "max", fmt_num, "n/a"),
    ("单核综合分 score", "cpu_single", "score", "max", fmt_num, "n/a"),
    ("多核 SHA256 (MiB/s)", "cpu_multi", "sha256_mibps", "max", fmt_num, "n/a"),
    ("多核素数筛 (primes/s)", "cpu_multi", "sieve_primes_per_sec", "max", fmt_num, "n/a"),
    ("多核综合分 score", "cpu_multi", "score", "max", fmt_num, "n/a"),
    ("并行效率", "cpu_multi", "parallel_efficiency", "max", fmt_pct, "n/a"),
    ("内存拷贝带宽 (MiB/s)", "memory", "copy_mibps", "max", fmt_num, "n/a"),
    ("numpy 内存带宽 (MiB/s)", "memory", "numpy_copy_mibps", "max", fmt_num, "n/a"),
    ("磁盘顺序写 (MiB/s)", "disk_seq", "write_mibps", "max", fmt_num, "n/a"),
    ("磁盘顺序读 (MiB/s)", "disk_seq", "read_mibps", "max", fmt_num, "n/a"),
    ("fsync 写 (MiB/s)", "disk_seq", "fsync_write_mibps", "max", fmt_num, "n/a"),
    ("4K 随机读 (IOPS)", "disk_rand", "r4k_read_iops", "max", fmt_num, "n/a"),
    ("4K 随机写 (IOPS)", "disk_rand", "r4k_write_iops", "max", fmt_num, "n/a"),
    ("DNS (ms)", "net", "dns_ms", "min", fmt_num, "未测试"),
    ("TCP connect (ms)", "net", "tcp_connect_ms", "min", fmt_num, "未测试"),
    ("TLS 握手 (ms)", "net", "tls_handshake_ms", "min", fmt_num, "未测试"),
    ("TTFB (ms)", "net", "ttfb_ms", "min", fmt_num, "未测试"),
    ("HTTP GET (MiB/s)", "net", "http_get_mibps", "max", fmt_num, "未测试"),
    ("sysbench CPU (events/s)", "native", "sysbench_cpu_events_per_sec", "max", fmt_num, "-"),
    ("sysbench 内存 (MiB/s)", "native", "sysbench_memory_mibs", "max", fmt_num, "-"),
    ("7z MIPS", "native", "7z_mips", "max", fmt_num, "-"),
]


def build_bench_table(platforms):
    rows = []
    champions = {}
    for label, section, key, champ, fmt, na_text in BENCH_ROWS:
        states = [bench_state(rec, section, key) for rec in platforms]
        cells = render_cells(platforms, states, champ=champ, fmt=fmt, na_text=na_text)
        if champ and any(c.startswith("**") for c in cells):
            for rec, c in zip(platforms, cells):
                if c.startswith("**"):
                    champions[label] = rec["name"]
        rows.append((label, cells))
    return render_table("指标", platforms, rows), champions


# ---------------------------------------------------------------------------
# 第 3 节：归一化每核性能
# ---------------------------------------------------------------------------


def per_core_data(platforms):
    """返回 [(name, score, logical, per_core)]，仅含有完整数据的平台。"""
    out = []
    for rec in platforms:
        st = bench_state(rec, "cpu_single", "score")
        cst = x_cores(
            rec, "logical_cores", "logical", "cores_logical", "vcpus", "cpu_count"
        )
        if st[0] == "num" and cst[0] == "num" and cst[1] and cst[1] > 0:
            score = float(st[1])
            cores = int(cst[1])
            out.append((rec["name"], score, cores, score / cores))
    return out


def build_per_core_section(platforms):
    lines = ["## 🏆 归一化每核性能", ""]
    lines.append(
        "用 `cpu_single.score ÷ 逻辑核数` 归一，揭示「核多但单核弱」的平台"
        "（例如 4 核 x64 vs 3 核 arm64 的真实单核效率差距）。"
    )
    lines.append("")
    data = per_core_data(platforms)
    if not data:
        lines.append("> 数据不足，无法排名（没有任何平台同时具备单核 score 与逻辑核数）。")
        return "\n".join(lines), [], {}
    data.sort(key=lambda t: t[3], reverse=True)
    lines.append("| 排名 | 平台 | 单核 score | 逻辑核 | 每核效率 (score/核) |")
    lines.append("|---|---|---|---|---|")
    for i, (name, score, cores, pc) in enumerate(data, 1):
        lines.append(
            "| {} | {} | {} | {} | {} |".format(i, esc(name), fmt_num(score), cores, fmt_num(pc))
        )
    excluded = [
        rec["name"] for rec in platforms if rec["name"] not in {d[0] for d in data}
    ]
    lines.append("")
    ranking = [d[0] for d in data]
    lines.append("**排名结论：每核效率 {}**".format(" > ".join(esc(r) for r in ranking)))
    if excluded:
        lines.append("")
        lines.append(
            "> 未参与排名（缺单核 score 或逻辑核数）：{}".format(
                ", ".join(esc(n) for n in excluded)
            )
        )
    per_core_map = {d[0]: d[3] for d in data}
    return "\n".join(lines), ranking, per_core_map


# ---------------------------------------------------------------------------
# 第 5 节：单平台完整报告链接
# ---------------------------------------------------------------------------


def build_links_section(platforms):
    lines = ["## 🔗 单平台完整报告", ""]
    lines.append("在 run 页面底部 **Artifacts** 面板下载 `probe-<os>` 即可获得该平台的完整 Markdown 报告、bench.json 与 pip-freeze.txt。")
    lines.append("")
    lines.append("| 平台 | Artifact 名 | report.md 大小 |")
    lines.append("|---|---|---|")
    for rec in platforms:
        size = rec.get("report_md_bytes")
        size_txt = "{:,} B".format(size) if isinstance(size, int) else "-"
        lines.append(
            "| {} | `probe-{}` | {} |".format(esc(rec["name"]), esc(rec["name"]), size_txt)
        )
    notes = error_notes(platforms)
    if notes:
        lines.append("")
        lines.extend(notes)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def build_summary(platforms, derived):
    parts = []
    parts.append("## 📊 跨平台硬件 / 性能对比")
    parts.append("")
    parts.append(
        "共收集到 **{} 个平台**的 probe 数据。缺失 / 失败项显示 `n/a`、`未测试`、`-` "
        "或错误标记，不会中断汇总。".format(len(platforms))
    )
    parts.append("")
    hw_table, hw_champs = build_hw_table(platforms)
    parts.append(hw_table)
    parts.append("")
    notes = error_notes(platforms)
    if notes:
        parts.extend(notes)
        parts.append("")
    parts.append("> 🏅 **加粗** = 该行冠军（容量/核数取最大）。`-` = 无 cgroup 限制或未提供。")
    parts.append("")

    parts.append("## ⚡ 基准成绩对比")
    parts.append("")
    bench_table, bench_champs = build_bench_table(platforms)
    parts.append(bench_table)
    parts.append("")
    parts.append(
        "> 🏅 **加粗** = 该行冠军（吞吐 / 分数 / IOPS 取最大，延迟 ms 取最小）。"
        "`未测试` = 该平台未启用网络基准（`--net`）；`-` = 原生工具（sysbench / 7z）不可用。"
    )
    parts.append("> ⚠️ 分数为相对指数，跨平台比较需留意 Python 版本与 vCPU 数差异。")
    parts.append("")

    per_core_md, ranking, per_core_map = build_per_core_section(platforms)
    parts.append(per_core_md)
    parts.append("")

    parts.append("## ⚠️ 可比性说明")
    parts.append("")
    for line in COMPARABILITY_NOTES.splitlines():
        parts.append(line)
    parts.append("")

    parts.append(build_links_section(platforms))
    parts.append("")

    derived = dict(derived)
    derived["per_core_ranking"] = ranking
    derived["per_core_score"] = per_core_map
    derived["champions"] = {"hardware": hw_champs, "benchmarks": bench_champs}
    return "\n".join(parts), derived


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="合并各平台 probe artifact 的 bench.json，生成跨平台对比 summary.md"
    )
    ap.add_argument("--root", required=True, help="download-artifact 落地根目录")
    ap.add_argument("--out", required=True, help="summary.md 输出路径")
    ap.add_argument(
        "--github-summary",
        action="store_true",
        help="同时把结果追加写入 $GITHUB_STEP_SUMMARY 指向的文件",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(
            "ERROR: --root 目录不存在或不是目录: {}".format(root),
            file=sys.stderr,
        )
        return 1

    found = discover(root)
    if not found:
        print(
            "ERROR: 在 {} 下没有找到任何 bench.json / bench-<platform>.json，"
            "无法生成对比报告（请确认 probe job 是否成功上传 artifact，"
            "以及 download-artifact 的 path 是否指向该目录）。".format(root),
            file=sys.stderr,
        )
        return 1

    platforms = []
    for name in sorted(found, key=platform_sort_key):
        rec = load_platform(name, found[name], root)
        if rec["error"]:
            print(
                "WARN: 平台 {} 的 bench.json 解析失败: {}".format(name, trunc(rec["error"], 80)),
                file=sys.stderr,
            )
        else:
            print("OK: 平台 {} <- {}".format(name, rec["source"]))
        platforms.append(rec)
    print("共 {} 个平台参与对比: {}".format(len(platforms), ", ".join(p["name"] for p in platforms)))

    summary_md, derived = build_summary(platforms, {})

    out_path = Path(args.out)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    out_path.write_text(summary_md + "\n", encoding="utf-8")
    print("已写入 {}".format(out_path))

    merged = {
        "generated_by": "merge_report.py",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "root": str(root),
        "platform_count": len(platforms),
        "platform_order": [p["name"] for p in platforms],
        "platforms": {
            p["name"]: {
                "source": p["source"],
                "report_md_bytes": p.get("report_md_bytes"),
                "error": p.get("error"),
                "bench": p.get("data"),
            }
            for p in platforms
        },
        "derived": derived,
    }
    merged_path = out_path.parent / "merged.json"
    merged_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("已写入 {}".format(merged_path))

    if args.github_summary:
        gs = os.environ.get("GITHUB_STEP_SUMMARY", "")
        if gs:
            try:
                with open(gs, "a", encoding="utf-8") as f:
                    f.write("\n\n" + summary_md + "\n")
                print("已追加写入 $GITHUB_STEP_SUMMARY -> {}".format(gs))
            except OSError as e:
                print("WARN: 写入 {} 失败: {}".format(gs, e), file=sys.stderr)
        else:
            print("WARN: --github-summary 已指定但 $GITHUB_STEP_SUMMARY 未设置，跳过", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
