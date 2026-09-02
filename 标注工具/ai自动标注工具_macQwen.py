# -*- coding: utf-8 -*-
"""
精子头部 AI 自动标注工具 v3（标注 + 核查双模式）
================================================================
v3 新增：
  1. 程序启动时选择工作模式：
     - AI自动标注模式（现有功能）：从空白开始标注
     - AI核查模式（新增功能）：基于已有坐标文件，AI修正错误坐标、补全遗漏精子
  2. 核查模式提示词与约束参考原标注模式，保证坐标精度
  3. 核查模式预览图用颜色区分：绿色=原有保留，蓝色=AI新增，红色=AI删除

用法：
    python ai自动标注工具.py
================================================================
"""
from __future__ import annotations

import base64
import os
import re
import sys
import time
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tkinter import filedialog

import cv2
import numpy as np
from openai import OpenAI

# ── 固定配置参数（Mac mini 本地 Qwen 模型）────────────────────────────
API_KEY = "Azd7f9C-_QmLOqNuNWbsS9XAjnjWy1rWtGpi6B62UIo"
EP_ID = "mtplx-qwen38-27b-optimized-speed"
BASE_URL = "http://xinmeitideMac-mini.local:8000/v1"

# 图片预期尺寸（仅作校验，实际以读取为准）
EXPECT_W, EXPECT_H = 1280, 1280
# 顶部黑边过滤阈值：yc小于该值丢弃，无黑边图片设为0.0
BLACK_TOP_THRESHOLD = 0.0
# 框宽高固定值（标准YOLO格式）
FIXED_BOX_W = 0.015625
FIXED_BOX_H = 0.015625

# 去重IOU阈值：超过则视为重复框
IOU_THRESHOLD = 0.3
# 目标数异常告警阈值
MIN_TARGETS = 5
MAX_TARGETS = 80

# 支持的图片扩展名
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# API调用配置：MAX_RETRIES 为外层退避重试次数
MAX_RETRIES = 4
# 全局最小请求间隔（秒）：并发下仍保证任意两次请求间隔不小于该值，防限流
GLOBAL_MIN_INTERVAL = 0.4

# 运行时由用户在启动阶段选择（默认按上次配置：核查模式 + 思考模式 + 30线程）
WORK_MODE = "review"     # annotate=自动标注  review=AI核查
THINK_MODE = True         # True=思考模式  False=快速模式
CONCURRENCY = 30          # 1~100

# 初始化客户端，长超时避免断开；max_retries=0 由本程序自行退避重试
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=1200.0,
    max_retries=0,
)

# GBK控制台下emoji会导致UnicodeEncodeError崩溃，替换为?而非中断
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")


# ── 全局打印锁与请求限流器 ────────────────────────────────────────────
print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    """多线程安全的打印：避免并发输出串行交错"""
    with print_lock:
        print(*args, **kwargs)


class RateLimiter:
    """全局请求限流：保证任意两次请求开始之间至少间隔 min_interval 秒"""

    def __init__(self, min_interval: float):
        self._lock = threading.Lock()
        self._last = 0.0
        self._min = min_interval

    def wait(self):
        with self._lock:
            now = time.monotonic()
            wait = self._last + self._min - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._last = now


rate_limiter = RateLimiter(GLOBAL_MIN_INTERVAL)


# ── 弹窗路径选择 ──────────────────────────────────────────────────────
def select_paths_annotate() -> tuple[Path, Path, Path | None]:
    """标注模式：选择输入图片目录、标签输出目录、预览图输出目录
    支持环境变量非交互模式：AI_INPUT, AI_OUT_LABEL, AI_DRAW(可选)"""
    # 环境变量非交互模式
    env_input = os.environ.get("AI_INPUT")
    env_out = os.environ.get("AI_OUT_LABEL")
    if env_input and env_out:
        print(f"📂 [环境变量模式] 输入图片目录: {env_input}")
        print(f"📂 [环境变量模式] 标签输出目录: {env_out}")
        env_draw = os.environ.get("AI_DRAW")
        draw_path = Path(env_draw) if env_draw else None
        if draw_path:
            print(f"📂 [环境变量模式] 预览图目录: {env_draw}")
        else:
            print("📂 [环境变量模式] 不生成预览图")
        return Path(env_input), Path(env_out), draw_path

    root = tk.Tk()
    root.withdraw()

    print("📂 请选择输入图片目录...")
    input_dir = filedialog.askdirectory(title="选择输入图片目录")
    if not input_dir:
        print("❌ 未选择输入目录，程序退出。")
        sys.exit(1)

    print("📂 请选择YOLO标签输出目录...")
    label_dir = filedialog.askdirectory(title="选择标签输出目录")
    if not label_dir:
        print("❌ 未选择标签输出目录，程序退出。")
        sys.exit(1)

    print("📂 请选择预览图输出目录（点取消则不生成预览图）...")
    draw_dir = filedialog.askdirectory(title="选择预览图输出目录（取消则不生成）")
    draw_dir_path = Path(draw_dir) if draw_dir else None

    return Path(input_dir), Path(label_dir), draw_dir_path


def select_paths_review() -> tuple[Path, Path, Path, Path | None]:
    """核查模式：选择输入图片目录、已有标签目录、修正后标签输出目录、预览图目录
    支持环境变量非交互模式：AI_INPUT, AI_SRC_LABEL, AI_OUT_LABEL, AI_DRAW(可选)"""
    # 环境变量非交互模式
    env_input = os.environ.get("AI_INPUT")
    env_src = os.environ.get("AI_SRC_LABEL")
    env_out = os.environ.get("AI_OUT_LABEL")
    if env_input and env_src and env_out:
        print(f"📂 [环境变量模式] 输入图片目录: {env_input}")
        print(f"📂 [环境变量模式] 已有标签目录: {env_src}")
        print(f"📂 [环境变量模式] 修正后标签目录: {env_out}")
        env_draw = os.environ.get("AI_DRAW")
        draw_path = Path(env_draw) if env_draw else None
        if draw_path:
            print(f"📂 [环境变量模式] 预览图目录: {env_draw}")
        else:
            print("📂 [环境变量模式] 不生成预览图")
        return Path(env_input), Path(env_src), Path(env_out), draw_path

    root = tk.Tk()
    root.withdraw()

    print("📂 请选择输入图片目录...")
    input_dir = filedialog.askdirectory(title="选择输入图片目录")
    if not input_dir:
        print("❌ 未选择输入目录，程序退出。")
        sys.exit(1)

    print("📂 请选择已有YOLO标签目录（待核查）...")
    src_label_dir = filedialog.askdirectory(title="选择已有YOLO标签目录（待核查）")
    if not src_label_dir:
        print("❌ 未选择已有标签目录，程序退出。")
        sys.exit(1)

    print("📂 请选择修正后标签输出目录（可与输入目录相同即覆写）...")
    out_label_dir = filedialog.askdirectory(title="选择修正后标签输出目录（可与输入相同即覆写）")
    if not out_label_dir:
        print("❌ 未选择输出目录，程序退出。")
        sys.exit(1)

    print("📂 请选择预览图输出目录（点取消则不生成预览图）...")
    draw_dir = filedialog.askdirectory(title="选择预览图输出目录（取消则不生成）")
    draw_dir_path = Path(draw_dir) if draw_dir else None

    return Path(input_dir), Path(src_label_dir), Path(out_label_dir), draw_dir_path


# ── 启动时交互选择 ────────────────────────────────────────────────────
def choose_work_mode() -> str:
    """选择工作模式：annotate=自动标注  review=AI核查
    支持环境变量 AI_MODE=annotate/review 非交互模式"""
    env_mode = os.environ.get("AI_MODE")
    if env_mode in ("annotate", "review"):
        mode_name = "AI自动标注" if env_mode == "annotate" else "AI核查"
        print(f"[环境变量模式] 工作模式: {mode_name}")
        return env_mode

    while True:
        raw = input("\n请选择工作模式：\n"
                    "  1 = AI自动标注模式（从空白开始标注，现有功能）\n"
                    "  2 = AI核查模式（基于已有坐标修正错误、补全遗漏，新增功能）\n"
                    "请输入 1 或 2（默认 1）：").strip()
        if not raw or raw == "1":
            return "annotate"
        if raw == "2":
            return "review"
        print("  输入无效，请输入 1 或 2。")


def choose_mode() -> bool:
    """选择标注模式：True=思考模式(慢·更准)  False=快速模式(快)
    支持环境变量 AI_THINK=1/0 非交互模式"""
    env_think = os.environ.get("AI_THINK")
    if env_think is not None:
        is_think = env_think in ("1", "true", "True", "yes")
        mode_name = "思考模式" if is_think else "快速模式"
        print(f"[环境变量模式] 标注模式: {mode_name}")
        return is_think

    while True:
        raw = input("\n请选择标注模式：\n"
                    "  1 = 思考模式（深度思考，更准，较慢）\n"
                    "  2 = 快速模式（响应快，速度优先）\n"
                    "请输入 1 或 2（默认 1）：").strip()
        if not raw or raw == "1":
            return True
        if raw == "2":
            return False
        print("  输入无效，请输入 1 或 2。")


def choose_threads() -> int:
    """选择并发线程数量（1~100）
    支持环境变量 AI_THREADS=N 非交互模式"""
    env_threads = os.environ.get("AI_THREADS")
    if env_threads is not None:
        try:
            n = int(env_threads)
            if 1 <= n <= 100:
                print(f"[环境变量模式] 并发线程: {n}")
                return n
        except ValueError:
            pass

    while True:
        raw = input("请选择并发线程数量（1~100，默认 1）：").strip()
        if not raw:
            return 1
        try:
            n = int(raw)
        except ValueError:
            print("  输入无效，请输入 1~100 之间的整数。")
            continue
        if 1 <= n <= 100:
            return n
        print("  输入无效，请输入 1~100 之间的整数。")


# ── 工具函数 ───────────────────────────────────────────────────────────
def natural_sort_key(path_str: str) -> list:
    name = Path(path_str).name.lower()
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", name)]


def append_error_log(log_path: Path, img_name: str, error_msg: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {img_name} | {error_msg}\n")


def read_existing_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """读取已有标签文件，返回 [(cls, xc, yc, w, h), ...]"""
    if not label_path.exists():
        return []
    try:
        content = label_path.read_text(encoding="utf-8").strip()
    except Exception:
        return []
    boxes = []
    for line in content.splitlines():
        parts = line.strip().replace(",", " ").split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except (ValueError, IndexError):
            continue
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            continue
        boxes.append((cls, xc, yc, w, h))
    return boxes


def format_existing_labels_for_prompt(boxes: list[tuple[int, float, float, float, float]],
                                        img_w: int, img_h: int) -> str:
    """将已有标签格式化为提示词中的文本，每行包含归一化坐标和像素坐标"""
    if not boxes:
        return "（无已有标注）"
    lines = []
    for i, (cls, xc, yc, w, h) in enumerate(boxes, 1):
        px = int(xc * img_w)
        py = int(yc * img_h)
        lines.append(f"  {i}. cls={cls} xc={xc:.6f} yc={yc:.6f} (像素位置: x={px}, y={py})")
    return "\n".join(lines)


# ── 提示词生成 ─────────────────────────────────────────────────────────
def generate_annotate_prompt(img_w: int, img_h: int) -> str:
    """自动标注模式提示词（原有功能）"""
    box_px = round(FIXED_BOX_W * img_w)
    return f"""你是一名专业的精子头部标注工程师，标注精度要求≥99.5%。请逐一枚举图片中所有精子头部，输出标准YOLO格式坐标。

【图像信息（最高优先级约束）】
- 当前图像尺寸固定为 {img_w}×{img_h} 像素，正方形，无黑边、无padding、无letterbox
- 你输出的所有坐标必须基于这幅原始 {img_w}×{img_h} 图像
- 禁止使用你内部缩放、补边、分块后的图像坐标系，必须在原始图幅上定位

【坐标系定义（必须严格遵守）】
- 坐标原点 (0,0) 位于图像左上角像素处
- x 轴向右增大，y 轴向下增大（y=0在最上方，y最大在最下方，禁止翻转）
- 四角锚点：左上=(0,0)，右上=(1,0)，左下=(0,1)，右下=(1,1)，图像正中心=(0.5,0.5)
- 换算公式：xc = 头部几何中心的像素x ÷ {img_w}；yc = 头部几何中心的像素y ÷ {img_h}
- 示例：头部中心在像素(160, 480) → xc=160/{img_w}={160/img_w:.6f}，yc=480/{img_h}={480/img_h:.6f}

【禁止事项】
- 禁止输出像素坐标（如 160 480）
- 禁止输出 0~1000 或 0~100 量纲的坐标
- 禁止输出框的左上角/右下角坐标，xc yc 必须是头部几何中心
- 禁止 y 轴翻转（原点在左上，不在左下）

【目标定义】
- 目标：精子头部，呈椭圆形/梨形，大小约{box_px}像素
- 有效目标包括：完整头部、边缘截断头部、略模糊头部、不同朝向头部、对比度偏低头部
- 无效目标：尾部、鞭毛、杂质、气泡、划痕、尘埃、聚集成团的非头部区域

【标注规则】
1. 每个精子头部只标注一个框，中心点严格落在头部几何中心，禁止偏移到尾部或边缘
2. 必须检出全部目标，优先保证召回率，禁止漏检；宁可重复不可遗漏
3. 禁止将杂质、噪点误判为精子头部
4. 同一个头部禁止重复标注

【推理步骤（输出前逐条自查）】
1. 先在原始 {img_w}×{img_h} 像素坐标系中确定每个头部中心的像素位置
2. 再将像素值分别除以 {img_w} 和 {img_h}，得到 0~1 归一化坐标
3. 自查：最靠左的精子 xc 应接近 0，最靠右的接近 1，最上方的 yc 接近 0，最下方的接近 1；若所有坐标挤在一小块区域，说明你用错了坐标系，必须重算

【输出格式】
每一行严格为：0 xc yc 0.015625 0.015625
- xc, yc：归一化中心点坐标，0到1之间，保留6位小数
- 框宽高固定为0.015625，不可修改

【输出要求】
- 仅输出YOLO标签行，不要任何解释、说明、开场白、结束语、代码块标记
- 按从上到下顺序输出，确保无遗漏
"""


def generate_review_prompt(img_w: int, img_h: int,
                            existing_boxes: list[tuple[int, float, float, float, float]]) -> str:
    """AI核查模式提示词：基于已有坐标修正错误、补全遗漏"""
    box_px = round(FIXED_BOX_W * img_w)
    existing_text = format_existing_labels_for_prompt(existing_boxes, img_w, img_h)
    if existing_boxes:
        existing_section = f"""【已有标注（待审核）】
共 {len(existing_boxes)} 个已有标注：
{existing_text}"""
    else:
        existing_section = f"""【已有标注（待审核）】
共 0 个已有标注（无已有坐标文件或文件为空）。

⚠️ 重要提示：此图片可能是空画面（无任何精子头部），也可能是遗漏了全部标注的含精子画面。
请你仔细扫描全图后判断：
- 如果画面中确实没有任何精子头部（仅有背景、杂质、气泡、划痕），判定为空画面，输出空内容（0行）
- 如果画面中存在精子头部，必须补全全部标注，且数量应 ≥ 20 个"""
    return f"""你是一名专业的精子头部标注审核工程师，审核精度要求≥99.5%。图片中已有初步标注坐标，请你审核并修正：删除错误标注、修正偏移坐标、补全遗漏的精子头部，输出完整的修正后YOLO格式坐标。

【图像信息（最高优先级约束）】
- 当前图像尺寸固定为 {img_w}×{img_h} 像素，正方形，无黑边、无padding、无letterbox
- 你输出的所有坐标必须基于这幅原始 {img_w}×{img_h} 图像
- 禁止使用你内部缩放、补边、分块后的图像坐标系，必须在原始图幅上定位

【坐标系定义（必须严格遵守）】
- 坐标原点 (0,0) 位于图像左上角像素处
- x 轴向右增大，y 轴向下增大（y=0在最上方，y最大在最下方，禁止翻转）
- 四角锚点：左上=(0,0)，右上=(1,0)，左下=(0,1)，右下=(1,1)，图像正中心=(0.5,0.5)
- 换算公式：xc = 头部几何中心的像素x ÷ {img_w}；yc = 头部几何中心的像素y ÷ {img_h}
- 示例：头部中心在像素(160, 480) → xc=160/{img_w}={160/img_w:.6f}，yc=480/{img_h}={480/img_h:.6f}

{existing_section}

【审核任务】
1. 检查每个已有标注是否准确：中心点是否落在精子头部几何中心，是否有偏移到尾部/边缘/杂质
2. 删除错误标注：标注在杂质、气泡、尾部、空白区域的框必须删除
3. 修正偏移标注：中心点偏移的框，修正到头部几何中心
4. 补全遗漏精子：图片中存在但未标注的精子头部必须补全
5. 保留正确标注：位置准确的框直接保留

【数据集先验约束（最高优先级，必须严格遵守）】
本数据集具有以下已知分布特征，审核时必须以此为先验判断依据：

1. **含精子画面的数量下限**：
   - 凡是画面中存在精子头部的图片，其精子头部数量必然 ≥ 20 个
   - 不会出现只有 1~19 个精子的稀疏画面
   - 因此，如果你审核后输出 1~19 个框，必须高度警惕：极有可能发生了严重漏检，应重新扫描全图补全遗漏
   - 只有两种合法输出数量：0 个（空画面）或 ≥ 20 个（含精子画面）

2. **空画面的存在与处理**：
   - 本数据集中存在空画面：即画面中没有任何精子头部，仅有背景纹理、杂质、气泡、划痕、尘埃等
   - 空画面的判断标准：仔细扫描全图，确认不存在任何椭圆形/梨形、大小约10像素的精子头部结构
   - 空画面必须输出空坐标文件（即输出内容为空，0行YOLO标签），不能输出任何框
   - 如果已有标签文件中存在标注，但画面实际为空画面，必须删除全部已有标注，输出空文件
   - 空画面是数据集的正常组成部分，输出空坐标文件是正确行为，不是漏检

3. **数量异常的自查流程**：
   - 输出前必须统计最终框的数量
   - 若数量为 0：确认画面确实为空画面（无任何精子头部），则输出空文件
   - 若数量为 1~19：判定为异常，必须重新全图扫描，找出遗漏的精子头部，直到数量 ≥ 20 或确认画面为空（数量=0）
   - 若数量 ≥ 20：正常输出

【禁止事项】
- 禁止输出像素坐标（如 160 480）
- 禁止输出 0~1000 或 0~100 量纲的坐标
- 禁止输出框的左上角/右下角坐标，xc yc 必须是头部几何中心
- 禁止 y 轴翻转（原点在左上，不在左下）
- 禁止保留错误标注或遗漏真实精子头部

【目标定义】
- 目标：精子头部，呈椭圆形/梨形，大小约{box_px}像素
- 有效目标包括：完整头部、边缘截断头部、略模糊头部、不同朝向头部、对比度偏低头部
- 无效目标：尾部、鞭毛、杂质、气泡、划痕、尘埃、聚集成团的非头部区域

【审核规则】
1. 每个精子头部只保留一个框，中心点严格落在头部几何中心
2. 必须检出全部真实目标，优先保证召回率，禁止漏检
3. 禁止将杂质、噪点误判为精子头部（已有标注中的错误框必须删除）
4. 同一个头部禁止重复标注
5. 输出的是修正后的完整坐标列表，不是增量（包含保留的+修正的+新增的全部框）

【推理步骤（输出前逐条自查）】
1. 先在原始 {img_w}×{img_h} 像素坐标系中，逐个核对已有标注的像素位置是否准确
2. 标记需要删除的错误框、需要修正的偏移框
3. 扫描全图，找出未标注的精子头部
4. 汇总：保留正确框 + 修正偏移框 + 新增遗漏框，删除错误框
5. 将所有框的像素坐标除以 {img_w} 和 {img_h}，得到 0~1 归一化坐标
6. 自查：最靠左的精子 xc 应接近 0，最靠右的接近 1，最上方的 yc 接近 0，最下方的接近 1

【输出格式】
每一行严格为：0 xc yc 0.015625 0.015625
- xc, yc：归一化中心点坐标，0到1之间，保留6位小数
- 框宽高固定为0.015625，不可修改

【输出要求】
- 仅输出修正后的完整YOLO标签行，不要任何解释、说明、开场白、结束语、代码块标记
- 按从上到下顺序输出，确保无遗漏
- 如果图片为空画面（无任何精子头部），输出空内容（0行），这是正确行为，不是漏检
- 含精子画面的输出数量必须 ≥ 20，1~19 个属于异常，需重新扫描补全
"""


# ── 图片读取与编码 ────────────────────────────────────────────────────
def read_image(file_path: str) -> tuple[np.ndarray, str]:
    data = np.fromfile(file_path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("无法读取图片，文件可能损坏或格式不支持")

    h, w = img.shape[:2]
    warning = ""
    if w != EXPECT_W or h != EXPECT_H:
        warning = f"尺寸 {w}x{h} 非预期 {EXPECT_W}x{EXPECT_H}"
    return img, warning


def img_to_base64(img: np.ndarray) -> str:
    """视觉无损压缩，不影响识别精度，减小传输体积"""
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buffer).decode("utf-8")


def is_label_valid(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    try:
        content = label_path.read_text(encoding="utf-8").strip()
        if not content:
            return False
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                return False
            float(parts[1])
            float(parts[2])
        return True
    except Exception:
        return False


# ── 模型调用（按所选模式 + 限流退避） ─────────────────────────────────
def call_model(b64_data: str, img_w: int, img_h: int, img_name: str = "",
                prompt: str | None = None) -> str:
    """按全局 THINK_MODE 调用模型，返回经防御性清洗的原始输出。
    若传入 prompt 则使用自定义提示词（核查模式），否则使用自动标注提示词"""
    PROGRESS_INTERVAL = 20.0
    if prompt is None:
        prompt = generate_annotate_prompt(img_w, img_h)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}}
            ]
        }
    ]

    use_thinking = THINK_MODE
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        rate_limiter.wait()
        try:
            extra_body = {"thinking": {"type": "enabled"}} if use_thinking \
                else {"thinking": {"type": "disabled"}}
            resp = client.chat.completions.create(
                model=EP_ID,
                temperature=0.0,
                stream=True,
                max_tokens=4096,
                extra_body=extra_body,
                messages=messages,
            )
            full_text = ""
            chunk_count = 0
            first_content = False
            req_start = time.monotonic()
            last_progress = req_start
            for chunk in resp:
                chunk_count += 1
                delta = chunk.choices[0].delta.content
                if delta is not None:
                    if not first_content:
                        first_content = True
                        safe_print(f"    ⏳ {img_name} 已开始接收响应...")
                    full_text += delta
                now = time.monotonic()
                if now - last_progress >= PROGRESS_INTERVAL:
                    safe_print(f"    ⏳ {img_name} 仍在处理中，已等待 {int(now - req_start)}s，"
                               f"已接收 {chunk_count} 个数据块 / {len(full_text)} 字")
                    last_progress = now
            safe_print(f"    ✅ {img_name} 响应接收完成：{chunk_count} 块 / {len(full_text)} 字")
            full_text = re.sub(r" thinking.*? response", "", full_text, flags=re.DOTALL)
            return full_text.strip()
        except Exception as e:
            last_error = e
            msg = str(e)
            is_rate_limit = ("429" in msg) or ("rate" in msg.lower()) \
                or ("too many" in msg.lower())
            is_thinking_err = ("thinking" in msg.lower()) or (
                "parameter" in msg.lower() and "think" in msg.lower())

            if is_thinking_err and use_thinking and attempt == 1:
                safe_print("  ⚠️ 端点不支持 thinking 参数，自动降级为不带 thinking 重试一次")
                use_thinking = False
                continue

            backoff = (6 if is_rate_limit else 2) * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                safe_print(f"\n  ⚠️ 第{attempt}次调用失败({type(e).__name__}: {msg[:120]})，"
                           f"{backoff}秒后重试...")
                time.sleep(backoff)
            else:
                raise last_error


# ── 输出清洗与IOU去重 ────────────────────────────────────────────────
def compute_iou(x1: float, y1: float, w1: float, h1: float,
                x2: float, y2: float, w2: float, h2: float) -> float:
    x1_min, x1_max = x1 - w1 / 2, x1 + w1 / 2
    y1_min, y1_max = y1 - h1 / 2, y1 + h1 / 2
    x2_min, x2_max = x2 - w2 / 2, x2 + w2 / 2
    y2_min, y2_max = y2 - h2 / 2, y2 + h2 / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def clean_yolo_output(text: str, img_w: int, img_h: int) -> list[str]:
    # 防御性清洗：思考过程、代码块标记、逗号分隔符统统处理掉
    text = re.sub(r" thinking.*? response", "", text, flags=re.DOTALL)
    text = text.replace("```", "").replace("yolo", "").strip()
    text = text.replace(",", " ")

    raw_boxes = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
        except (ValueError, IndexError):
            continue
        raw_boxes.append((cls, xc, yc))

    if not raw_boxes:
        return []

    # ── 坐标量纲自动识别 ──
    over = sum(1 for _, x, y in raw_boxes if x > 1.05 or y > 1.05)
    if raw_boxes and over > len(raw_boxes) * 0.5:
        max_val = max(max(x, y) for _, x, y in raw_boxes)
        if max_val > max(img_w, img_h) + 5:
            raw_boxes = [(c, x / 1000.0, y / 1000.0) for c, x, y in raw_boxes]
            scale_note = "0~1000"
        else:
            raw_boxes = [(c, x / img_w, y / img_h) for c, x, y in raw_boxes]
            scale_note = "像素"
        safe_print(f"\n  ⚠️ 检测到模型输出了{scale_note}量纲坐标，已自动换算为0~1归一化坐标")

    filtered = []
    for cls, xc, yc in raw_boxes:
        if not (-0.05 <= xc <= 1.05 and -0.05 <= yc <= 1.05):
            continue
        if yc < BLACK_TOP_THRESHOLD:
            continue
        if xc <= 0.0 and yc <= 0.0:
            continue
        filtered.append((cls, min(max(xc, 0.0), 1.0), min(max(yc, 0.0), 1.0)))

    # IOU去重
    keep_boxes = []
    filtered.sort(key=lambda x: x[2])

    for cls, xc, yc in filtered:
        duplicate = False
        for k_cls, k_xc, k_yc in keep_boxes:
            if cls != k_cls:
                continue
            iou = compute_iou(xc, yc, FIXED_BOX_W, FIXED_BOX_H,
                              k_xc, k_yc, FIXED_BOX_W, FIXED_BOX_H)
            if iou > IOU_THRESHOLD:
                duplicate = True
                break
        if not duplicate:
            keep_boxes.append((cls, xc, yc))

    valid_lines = []
    for cls, xc, yc in keep_boxes:
        valid_lines.append(
            f"{cls} {xc:.6f} {yc:.6f} {FIXED_BOX_W:.6f} {FIXED_BOX_H:.6f}"
        )

    return valid_lines


# ── 绘制预览图 ─────────────────────────────────────────────────────────
def draw_yolo_box(img: np.ndarray, yolo_lines: list[str],
                  color: tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    vis = img.copy()
    h_img, w_img = vis.shape[:2]

    for line in yolo_lines:
        parts = line.split()
        _, xc, yc, bw, bh = map(float, parts)

        cx = int(xc * w_img)
        cy = int(yc * h_img)
        box_w = int(bw * w_img)
        box_h = int(bh * h_img)

        x1 = max(0, cx - box_w // 2)
        y1 = max(0, cy - box_h // 2)
        x2 = min(w_img - 1, cx + box_w // 2)
        y2 = min(h_img - 1, cy + box_h // 2)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    return vis


def draw_review_box(img: np.ndarray, final_lines: list[str],
                    existing_boxes: list[tuple[int, float, float, float, float]],
                    img_w: int, img_h: int) -> np.ndarray:
    """核查模式预览图：绿色=原有保留，蓝色=AI新增，红色=AI删除"""
    vis = img.copy()

    # 解析最终框的坐标集合
    final_coords = set()
    for line in final_lines:
        parts = line.split()
        if len(parts) >= 3:
            xc, yc = float(parts[1]), float(parts[2])
            final_coords.add((round(xc, 4), round(yc, 4)))

    # 红色：AI删除的原有框（在已有中但不在最终中）
    for cls, xc, yc, w, h in existing_boxes:
        cx = int(xc * img_w)
        cy = int(yc * img_h)
        bw = int(w * img_w)
        bh = int(h * img_h)
        x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
        x2, y2 = min(img_w - 1, cx + bw // 2), min(img_h - 1, cy + bh // 2)
        # 判断是否被保留（与最终框IOU>0.3视为保留）
        kept = False
        for fxc, fyc in final_coords:
            iou = compute_iou(xc, yc, w, h, fxc, fyc, FIXED_BOX_W, FIXED_BOX_H)
            if iou > 0.3:
                kept = True
                break
        color = (0, 200, 0) if kept else (0, 0, 255)  # 绿=保留，红=删除
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    # 蓝色：AI新增的框（在最终中但不与已有框重叠）
    for line in final_lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        xc, yc = float(parts[1]), float(parts[2])
        is_new = True
        for cls, exc, eyc, ew, eh in existing_boxes:
            iou = compute_iou(xc, yc, FIXED_BOX_W, FIXED_BOX_H,
                              exc, eyc, ew, eh)
            if iou > 0.3:
                is_new = False
                break
        if is_new:
            cx = int(xc * img_w)
            cy = int(yc * img_h)
            bw = int(FIXED_BOX_W * img_w)
            bh = int(FIXED_BOX_H * img_h)
            x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
            x2, y2 = min(img_w - 1, cx + bw // 2), min(img_h - 1, cy + bh // 2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 150, 0), 2)  # 蓝色=新增

    return vis


def save_image(img: np.ndarray, path: Path) -> None:
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(str(path))
    else:
        raise IOError("保存图片失败")


# ── 主流程：自动标注模式 ──────────────────────────────────────────────
def collect_images(input_dir: Path) -> list[Path]:
    images = []
    for f in input_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
            images.append(f)
    images.sort(key=lambda p: natural_sort_key(str(p)))
    return images


def process_single_annotate(img_path: Path, label_dir: Path, draw_dir: Path | None,
                            error_log_path: Path) -> tuple[bool, int, str]:
    """自动标注模式：单张图片处理"""
    stem = img_path.stem
    label_path = label_dir / f"{stem}.txt"

    if is_label_valid(label_path):
        status = "标签已存在且有效，跳过"
        if draw_dir is not None:
            try:
                img, _ = read_image(str(img_path))
                lines = [ln for ln in label_path.read_text(encoding="utf-8").splitlines()
                         if len(ln.split()) == 5]
                vis = draw_yolo_box(img, lines)
                save_image(vis, draw_dir / f"{stem}_draw.png")
                status += "（预览图已重绘）"
            except Exception as e:
                status += f"（预览图重绘失败: {e}）"
        return True, -1, status

    try:
        img, warning = read_image(str(img_path))
        h_img, w_img = img.shape[:2]

        b64_data = img_to_base64(img)
        safe_print(f"  📤 已向模型发送: {img_path.name}")
        raw_output = call_model(b64_data, w_img, h_img, img_path.name)
        valid_lines = clean_yolo_output(raw_output, w_img, h_img)

        content = "\n".join(valid_lines)
        if valid_lines:
            content += "\n"
        label_path.write_text(content, encoding="utf-8")

        if draw_dir is not None:
            draw_path = draw_dir / f"{stem}_draw.png"
            vis = draw_yolo_box(img, valid_lines)
            save_image(vis, draw_path)

        num = len(valid_lines)
        status = f"检测到 {num} 个目标"
        if num < MIN_TARGETS or num > MAX_TARGETS:
            status += " ⚠️数量异常"
            append_error_log(error_log_path, stem, f"目标数量异常: {num}")
        if warning:
            status += f"（{warning}）"

        return True, num, status

    except Exception as e:
        append_error_log(error_log_path, stem, str(e))
        return False, 0, f"失败: {e}"


# ── 主流程：AI核查模式 ────────────────────────────────────────────────
def process_single_review(img_path: Path, src_label_dir: Path, out_label_dir: Path,
                          draw_dir: Path | None,
                          error_log_path: Path) -> tuple[bool, int, str]:
    """AI核查模式：单张图片处理，基于已有坐标修正并补全"""
    stem = img_path.stem
    src_label_path = src_label_dir / f"{stem}.txt"
    out_label_path = out_label_dir / f"{stem}.txt"

    # 读取已有标签（文件不存在或为空均视为空标签）
    existing_boxes = read_existing_labels(src_label_path)
    has_existing_file = src_label_path.exists()
    if not existing_boxes:
        if has_existing_file:
            safe_print(f"  ⚠️ {img_path.name}: 已有标签文件为空，将判断是否为空画面")
        else:
            safe_print(f"  ⚠️ {img_path.name}: 无已有标签文件，将判断是否为空画面或补全标注")

    try:
        img, warning = read_image(str(img_path))
        h_img, w_img = img.shape[:2]

        b64_data = img_to_base64(img)
        if existing_boxes:
            safe_print(f"  🔍 核查中: {img_path.name} (已有{len(existing_boxes)}个标注)")
        else:
            safe_print(f"  🔍 核查中: {img_path.name} (无已有标注，判断空画面/补全)")

        # 生成核查提示词（包含已有坐标，空标签时提示判断空画面）
        review_prompt = generate_review_prompt(w_img, h_img, existing_boxes)
        raw_output = call_model(b64_data, w_img, h_img, img_path.name, prompt=review_prompt)
        valid_lines = clean_yolo_output(raw_output, w_img, h_img)

        # 保存修正后的标签（包括空文件）
        out_label_dir.mkdir(parents=True, exist_ok=True)
        content = "\n".join(valid_lines)
        if valid_lines:
            content += "\n"
        out_label_path.write_text(content, encoding="utf-8")

        # 生成预览图（颜色区分）
        if draw_dir is not None:
            draw_path = draw_dir / f"{stem}_review.png"
            vis = draw_review_box(img, valid_lines, existing_boxes, w_img, h_img)
            save_image(vis, draw_path)

        num_final = len(valid_lines)

        # 状态与异常判断（基于数据集先验：0=空画面正常，1~19=异常告警，≥20=正常）
        if num_final == 0:
            status = "核查完成: 判定为空画面，输出空坐标文件"
        elif num_final < 20:
            num_added = max(0, num_final - len(existing_boxes))
            num_removed = max(0, len(existing_boxes) - num_final)
            status = (f"核查完成: 原有{len(existing_boxes)} → 最终{num_final} "
                     f"(新增{num_added}, 删除{num_removed}) ⚠️数量异常(<20，可能漏检)")
            append_error_log(error_log_path, stem,
                           f"核查后数量异常(<20): {num_final}，可能存在漏检或非空画面误判为空")
        else:
            num_added = max(0, num_final - len(existing_boxes))
            num_removed = max(0, len(existing_boxes) - num_final)
            status = f"核查完成: 原有{len(existing_boxes)} → 最终{num_final} (新增{num_added}, 删除{num_removed})"

        if warning:
            status += f"（{warning}）"

        return True, num_final, status

    except Exception as e:
        append_error_log(error_log_path, stem, str(e))
        return False, 0, f"失败: {e}"


# ── 主函数 ─────────────────────────────────────────────────────────────
def main() -> int:
    global WORK_MODE, THINK_MODE, CONCURRENCY

    # 第一步：选择工作模式
    WORK_MODE = choose_work_mode()

    if WORK_MODE == "annotate":
        input_dir, label_dir, draw_dir = select_paths_annotate()
        error_log_path = label_dir.parent / "error_log.txt"
        stats_path = label_dir.parent / "dataset_stats.txt"
    else:
        input_dir, src_label_dir, out_label_dir, draw_dir = select_paths_review()
        error_log_path = out_label_dir.parent / "review_error_log.txt"
        stats_path = out_label_dir.parent / "review_stats.txt"

    if not input_dir.is_dir():
        print(f"❌ 输入目录不存在: {input_dir}")
        return 1

    # 第二步：选择标注模式和并发数
    THINK_MODE = choose_mode()
    CONCURRENCY = choose_threads()

    if WORK_MODE == "annotate":
        label_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_label_dir.mkdir(parents=True, exist_ok=True)
    if draw_dir is not None:
        draw_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_dir)
    if not images:
        print(f"❌ 目录中没有找到图片: {input_dir}")
        return 1

    mode_name = "AI自动标注" if WORK_MODE == "annotate" else "AI核查"
    print(f"\n{'='*60}")
    print(f"  {mode_name}【并发 {CONCURRENCY} 线程 · {'思考模式' if THINK_MODE else '快速模式'} · 限流退避】")
    print(f"{'='*60}")
    print(f"  输入图片目录: {input_dir}")
    if WORK_MODE == "annotate":
        print(f"  标签输出目录: {label_dir}")
    else:
        print(f"  已有标签目录: {src_label_dir}")
        print(f"  修正后标签目录: {out_label_dir}")
    print(f"  预览图目录: {draw_dir if draw_dir else '不生成'}")
    print(f"  图片数量: {len(images)}")
    print(f"  预期尺寸: {EXPECT_W}x{EXPECT_H}")
    print(f"  并发线程: {CONCURRENCY}")
    print(f"  标注模式: {'思考模式（深度思考·精度优先）' if THINK_MODE else '快速模式（响应更快）'}")
    print(f"  限流保护: 全局最小间隔 {GLOBAL_MIN_INTERVAL}s + 指数退避重试")
    print(f"  坐标防御: 量纲自动识别（0~1 / 0~1000 / 像素）+ 越界截断")
    if WORK_MODE == "review":
        print(f"  核查预览: 绿色=原有保留 蓝色=AI新增 红色=AI删除")
    print(f"{'='*60}\n")

    # 线程安全统计
    counters = {"success": 0, "skip": 0, "fail": 0, "detected": 0}
    num_list = []
    done = {"n": 0}
    stats_lock = threading.Lock()

    def handle_result(img_path: Path, ok: bool, detected: int, status: str) -> None:
        with stats_lock:
            if ok:
                if detected == -1:
                    counters["skip"] += 1
                else:
                    counters["success"] += 1
                    counters["detected"] += detected
                    num_list.append(detected)
            else:
                counters["fail"] += 1
            done["n"] += 1
            cur = done["n"]
        if ok and detected == -1:
            icon = "⏭️"
        elif ok:
            icon = "✅"
        else:
            icon = "❌"
        safe_print(f"[{cur:>4}/{len(images)}] {img_path.stem} {icon} {status}  "
                   f"[{time.strftime('%H:%M:%S')}]")

    # ── 并发执行 ──
    try:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            if WORK_MODE == "annotate":
                future_map = {
                    pool.submit(process_single_annotate, p, label_dir, draw_dir, error_log_path): p
                    for p in images
                }
            else:
                future_map = {
                    pool.submit(process_single_review, p, src_label_dir, out_label_dir,
                                draw_dir, error_log_path): p
                    for p in images
                }
            for fut in as_completed(future_map):
                p = future_map[fut]
                try:
                    ok, detected, status = fut.result()
                except Exception as e:
                    ok, detected, status = False, 0, f"线程异常: {e}"
                handle_result(p, ok, detected, status)
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断，已停止提交新任务，已完成的标签保留。")
        return 1

    num_list_sorted = sorted(num_list)
    avg_num = (counters["detected"] / len(num_list_sorted)
               if num_list_sorted else 0.0)
    min_num = num_list_sorted[0] if num_list_sorted else 0
    max_num = num_list_sorted[-1] if num_list_sorted else 0

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"{mode_name}统计\n")
        f.write(f"{'='*40}\n")
        f.write(f"总图片数: {len(images)}\n")
        f.write(f"成功处理: {counters['success']} 张\n")
        f.write(f"跳过(无标签/已存在): {counters['skip']} 张\n")
        f.write(f"失败: {counters['fail']} 张\n")
        f.write(f"总检测目标数: {counters['detected']}\n")
        f.write(f"单张平均目标数: {avg_num:.2f}\n")
        f.write(f"单张最少: {min_num}\n")
        f.write(f"单张最多: {max_num}\n")
        f.write(f"并发线程: {CONCURRENCY}\n")
        f.write(f"标注模式: {'思考模式' if THINK_MODE else '快速模式'}\n")
        f.write(f"工作模式: {mode_name}\n")
        if WORK_MODE == "annotate":
            f.write(f"\n标签目录: {label_dir}\n")
        else:
            f.write(f"\n已有标签目录: {src_label_dir}\n")
            f.write(f"修正后标签目录: {out_label_dir}\n")

    print(f"\n{'='*60}")
    print(f"  处理完成！【{mode_name} · 并发 {CONCURRENCY} 线程 · {'思考模式' if THINK_MODE else '快速模式'}】")
    print(f"   成功: {counters['success']} 张")
    print(f"   跳过: {counters['skip']} 张")
    print(f"   失败: {counters['fail']} 张")
    print(f"   总计检测到: {counters['detected']} 个目标")
    print(f"   单张平均: {avg_num:.2f} 个")
    if WORK_MODE == "annotate":
        print(f"   标签目录: {label_dir}")
    else:
        print(f"   已有标签目录: {src_label_dir}")
        print(f"   修正后标签目录: {out_label_dir}")
    if draw_dir:
        print(f"   预览图目录: {draw_dir}")
    print(f"   错误日志: {error_log_path}")
    print(f"   统计文件: {stats_path}")
    print(f"{'='*60}")

    return 0 if counters["fail"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
