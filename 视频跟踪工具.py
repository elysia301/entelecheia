# -*- coding: utf-8 -*-
"""
视频推理与目标跟踪系统
基于 PoleMos600 (YOLOv8s, 1280分辨率) 模型

功能：
  1. 导入视频进行推理
  2. ByteTrack / TrackTrack 目标跟踪（官方 ultralytics 实现）
  3. 自由调节跟踪宽容度（实时生效，无需重启跟踪）
  4. 支持目标消失后重新出现仍保持同一ID
     - ByteTrack: 通过 track_buffer（丢失轨迹保留帧数）+ match_thresh（匹配距离阈值）
     - TrackTrack: 额外支持 lost_match_thr（长遮挡松散重绑定，CVPR 2025）

宽容度参数说明：
  - track_buffer（缓冲帧数）: 目标消失后保留轨迹ID的帧数。
    值越大，目标消失越久仍可重识别。例如 track_buffer=200 表示目标消失200帧内重新出现仍保持同一ID。
  - match_thresh（匹配阈值）: 帧间匹配的距离阈值（0=完美匹配，1=无重叠）。
    值越大越宽容（允许IoU更低的匹配），值越小越严格。
  - track_high_thresh（高置信阈值）: 第一阶段匹配使用的检测置信度下限。
  - track_low_thresh（低置信阈值）: 第二阶段匹配使用的检测置信度下限。
  - new_track_thresh（新轨迹阈值）: 未匹配检测框创建新轨迹的置信度下限。
  - lost_match_thr（丢失重绑定阈值, 仅TrackTrack）: 松散重绑定门控。
    0=禁用; 设为 >match_thresh 的值（如0.9）可对长时间遮挡的目标进行松散二次重绑定。
"""
from __future__ import annotations

import sys
import os
import cv2
import time
import uuid
import shutil
import tempfile
import threading
import colorsys
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── 配置 ──────────────────────────────────────────────────────────────
MODEL_PATH = r"C:\Users\86137\Desktop\entelecheia\Demiurge\runs\train\PoleMos600\weights\best.pt"
DEFAULT_IMGSZ = 1280
DEFAULT_CONF = 0.05       # YOLO 检测置信度（低值，让 ByteTrack 两阶段匹配有足够低分检测）
DEFAULT_IOU = 0.7         # NMS IoU 阈值
IMGSZ_OPTIONS = [640, 960, 1280]

# ── 临时目录 ──
TEMP_DIR = Path(tempfile.gettempdir()) / "polemos600_track"
TEMP_VIDEO_DIR = TEMP_DIR / "videos"

# ── 宽容度预设 ──
# match_thresh 是距离阈值（1-IoU），值越大越宽容
TOLERANCE_PRESETS = {
    "保守": {
        "track_buffer": 30,
        "match_thresh": 0.5,
        "track_high_thresh": 0.5,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.5,
    },
    "标准": {
        "track_buffer": 80,
        "match_thresh": 0.8,
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.1,
        "new_track_thresh": 0.25,
    },
    "宽容": {
        "track_buffer": 200,
        "match_thresh": 0.9,
        "track_high_thresh": 0.2,
        "track_low_thresh": 0.05,
        "new_track_thresh": 0.2,
    },
    "极致宽容": {
        "track_buffer": 500,
        "match_thresh": 0.95,
        "track_high_thresh": 0.15,
        "track_low_thresh": 0.01,
        "new_track_thresh": 0.15,
    },
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


# ── 工具函数 ──────────────────────────────────────────────────────────
def has_non_ascii(path: str) -> bool:
    """检查路径是否包含非ASCII字符（中文等）"""
    try:
        path.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def get_track_color(track_id: int) -> tuple:
    """根据 track ID 生成视觉上可区分的颜色（BGR 格式）"""
    hue = (track_id * 0.618033988749895) % 1.0  # 黄金比例分布
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def generate_tracker_yaml(params: dict, tracker_type: str = "bytetrack") -> str:
    """
    根据当前参数生成跟踪器 YAML 配置文件。
    返回临时 YAML 文件路径（纯英文路径，避免中文路径问题）。
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = TEMP_DIR / f"tracker_{tracker_type}.yaml"

    if tracker_type == "bytetrack":
        config = {
            "tracker_type": "bytetrack",
            "track_high_thresh": params["track_high_thresh"],
            "track_low_thresh": params["track_low_thresh"],
            "new_track_thresh": params["new_track_thresh"],
            "track_buffer": params["track_buffer"],
            "match_thresh": params["match_thresh"],
            "fuse_score": True,
        }
    elif tracker_type == "tracktrack":
        config = {
            "tracker_type": "tracktrack",
            "track_high_thresh": params["track_high_thresh"],
            "track_low_thresh": params["track_low_thresh"],
            "new_track_thresh": params["new_track_thresh"],
            "track_buffer": params["track_buffer"],
            "match_thresh": params["match_thresh"],
            "lost_match_thr": params.get("lost_match_thr", 0.0),
            "iou_weight": 0.5,
            "reid_weight": 0.5,
            "conf_weight": 0.1,
            "angle_weight": 0.05,
            "penalty_p": 0.2,
            "penalty_q": 0.4,
            "reduce_step": 0.05,
            "tai_thr": 0.55,
            "min_track_len": 3,
            "gmc_method": "none",
            "with_reid": False,
            "model": "auto",
        }
    else:
        raise ValueError(f"不支持的跟踪器类型: {tracker_type}")

    # 手动写入 YAML（避免依赖 PyYAML）
    lines = [f"# 自动生成的 {tracker_type} 跟踪器配置"]
    for key, val in config.items():
        if isinstance(val, bool):
            val = "True" if val else "False"
        lines.append(f"{key}: {val}")
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    return str(yaml_path)


# ═══════════════════════════════════════════════════════════════════════
#  主应用
# ═══════════════════════════════════════════════════════════════════════
class VideoTrackerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PoleMos600 视频推理与目标跟踪系统")
        self.root.geometry("1320x780")
        self.root.minsize(1100, 650)

        # ── 状态 ──
        self.model = None
        self.model_loaded = False
        self.model_path = MODEL_PATH

        self.cap = None
        self.video_path = None
        self.temp_video_path = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.fps = 30
        self.orig_w = 0
        self.orig_h = 0

        # 处理线程
        self.worker_thread = None
        self.is_playing = False
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()  # 初始非暂停

        # 最新帧
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # 跟踪状态
        self.tracker_initialized = False
        self.tracker_type = "bytetrack"  # "bytetrack" or "tracktrack"
        self.active_ids = set()
        self.all_ids = set()
        self.id_first_frame = {}   # track_id -> 首次出现帧
        self.id_last_frame = {}    # track_id -> 最后出现帧
        self.tracker_yaml_path = None

        # 视频保存
        self.is_saving = False
        self.writer = None

        # 构建 UI
        self._setup_ui()

        # 开始帧轮询
        self._poll_frame()

        # 尝试加载模型
        self._try_load_model()

    # ── UI 构建 ──────────────────────────────────────────────────────
    def _setup_ui(self):
        """构建完整 UI 界面"""
        # 主容器
        main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        main_paned.pack(fill="both", expand=True, padx=4, pady=4)

        # ═══ 左侧控制面板 ═══
        left_frame = ttk.Frame(main_paned, width=320)
        main_paned.add(left_frame, weight=1)

        # 滚动容器
        canvas = tk.Canvas(left_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        self.scroll_frame = ttk.Frame(canvas)
        self.scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        sf = self.scroll_frame

        # ── 视频导入 ──
        ttk.Label(sf, text="── 视频导入 ──", font=("", 10, "bold")).pack(anchor="w", pady=(8, 2))
        ttk.Button(sf, text="📁 选择视频文件", command=self._import_video).pack(fill="x", pady=2)
        self.video_label = ttk.Label(sf, text="未导入视频", foreground="gray")
        self.video_label.pack(anchor="w", pady=2)

        # ── 模型 ──
        ttk.Label(sf, text="── 模型 ──", font=("", 10, "bold")).pack(anchor="w", pady=(10, 2))
        self.model_label = ttk.Label(sf, text="加载中...", foreground="gray")
        self.model_label.pack(anchor="w", pady=2)
        ttk.Button(sf, text="🔄 选择模型权重文件", command=self._select_model).pack(fill="x", pady=2)

        # ── 推理设置 ──
        ttk.Label(sf, text="── 推理设置 ──", font=("", 10, "bold")).pack(anchor="w", pady=(10, 2))

        conf_frame = ttk.Frame(sf)
        conf_frame.pack(fill="x", pady=2)
        ttk.Label(conf_frame, text="检测置信度:", width=12).pack(side="left")
        self.conf_var = tk.DoubleVar(value=DEFAULT_CONF)
        self.conf_spin = ttk.Spinbox(conf_frame, from_=0.01, to=1.0, increment=0.01,
                                     textvariable=self.conf_var, width=6)
        self.conf_spin.pack(side="right")

        iou_frame = ttk.Frame(sf)
        iou_frame.pack(fill="x", pady=2)
        ttk.Label(iou_frame, text="NMS IoU:", width=12).pack(side="left")
        self.iou_var = tk.DoubleVar(value=DEFAULT_IOU)
        self.iou_spin = ttk.Spinbox(iou_frame, from_=0.1, to=1.0, increment=0.05,
                                    textvariable=self.iou_var, width=6)
        self.iou_spin.pack(side="right")

        imgsz_frame = ttk.Frame(sf)
        imgsz_frame.pack(fill="x", pady=2)
        ttk.Label(imgsz_frame, text="推理分辨率:", width=12).pack(side="left")
        self.imgsz_var = tk.IntVar(value=DEFAULT_IMGSZ)
        imgsz_combo = ttk.Combobox(imgsz_frame, textvariable=self.imgsz_var, values=IMGSZ_OPTIONS,
                                    width=6, state="readonly")
        imgsz_combo.pack(side="right")
        imgsz_combo.bind("<<ComboboxSelected>>", lambda e: self._on_imgsz_change())

        # ── 跟踪设置 ──
        ttk.Label(sf, text="── 跟踪设置 ──", font=("", 10, "bold")).pack(anchor="w", pady=(10, 2))

        # 跟踪器类型
        tracker_frame = ttk.Frame(sf)
        tracker_frame.pack(fill="x", pady=2)
        ttk.Label(tracker_frame, text="跟踪器:", width=12).pack(side="left")
        self.tracker_type_var = tk.StringVar(value="bytetrack")
        tracker_combo = ttk.Combobox(tracker_frame, textvariable=self.tracker_type_var,
                                      values=["bytetrack", "tracktrack"],
                                      width=12, state="readonly")
        tracker_combo.pack(side="right")
        tracker_combo.bind("<<ComboboxSelected>>", lambda e: self._on_tracker_type_change())

        # 宽容度预设
        preset_frame = ttk.Frame(sf)
        preset_frame.pack(fill="x", pady=2)
        ttk.Label(preset_frame, text="宽容度:", width=12).pack(side="left")
        self.preset_var = tk.StringVar(value="标准")
        preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                     values=list(TOLERANCE_PRESETS.keys()),
                                     width=12, state="readonly")
        preset_combo.pack(side="right")
        preset_combo.bind("<<ComboboxSelected>>", lambda e: self._on_preset_change())

        # 参数滑块（实时生效）
        self._add_slider(sf, "缓冲帧数\n(丢失保留)", "track_buffer", 30, 1000, 80, 10)
        self._add_slider(sf, "匹配阈值\n(越大越宽容)", "match_thresh", 0.1, 0.99, 0.8, 0.01)
        self._add_slider(sf, "高置信阈值", "track_high_thresh", 0.05, 0.8, 0.25, 0.01)
        self._add_slider(sf, "低置信阈值", "track_low_thresh", 0.01, 0.5, 0.1, 0.01)
        self._add_slider(sf, "新轨迹阈值", "new_track_thresh", 0.05, 0.8, 0.25, 0.01)

        # lost_match_thr（仅 TrackTrack）
        self.lost_match_frame = ttk.Frame(sf)
        self.lost_match_frame.pack(fill="x", pady=2)
        ttk.Label(self.lost_match_frame, text="丢失重绑定\n(仅TrackTrack)", width=12, justify="left").pack(side="left")
        self.lost_match_thr_var = tk.DoubleVar(value=0.0)
        lost_scale = ttk.Scale(self.lost_match_frame, from_=0.0, to=0.99,
                               variable=self.lost_match_thr_var, length=80,
                               command=lambda v: self._on_slider_change("lost_match_thr"))
        lost_scale.pack(side="left", padx=4)
        self.lost_match_thr_label = ttk.Label(self.lost_match_frame, text="0.00", width=5)
        self.lost_match_thr_label.pack(side="right")
        self.lost_match_frame.pack_forget()  # 默认隐藏（ByteTrack 模式）

        # 重置跟踪按钮
        self.reset_track_btn = ttk.Button(sf, text="🔄 重置跟踪（从头开始）", command=self._reset_tracking)
        self.reset_track_btn.pack(fill="x", pady=(6, 2))

        # 参数说明
        ttk.Label(sf, text="提示: 参数修改实时生效（下一帧应用）。\n更换跟踪器类型需重置跟踪。",
                  foreground="gray", font=("", 8)).pack(anchor="w", pady=4)

        # ── 输出 ──
        ttk.Label(sf, text="── 输出 ──", font=("", 10, "bold")).pack(anchor="w", pady=(10, 2))
        ttk.Button(sf, text="💾 保存跟踪视频", command=self._save_video).pack(fill="x", pady=2)
        ttk.Button(sf, text="🗑 清理临时缓存", command=self._clear_cache).pack(fill="x", pady=2)

        # ── 统计 ──
        ttk.Label(sf, text="── 跟踪统计 ──", font=("", 10, "bold")).pack(anchor="w", pady=(10, 2))
        self.stats_label = ttk.Label(sf, text="当前ID: 0\n总ID: 0\n最长寿命: 0帧",
                                     justify="left", font=("Consolas", 9))
        self.stats_label.pack(anchor="w", pady=2)

        # ═══ 右侧显示区 ═══
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=3)

        # 视频画布
        self.canvas = tk.Canvas(right_frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.bind("<Configure>", lambda e: self._on_canvas_resize())

        # 进度条
        progress_frame = ttk.Frame(right_frame)
        progress_frame.pack(fill="x", padx=4, pady=2)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Scale(progress_frame, from_=0, to=100,
                                       variable=self.progress_var,
                                       command=self._on_progress_seek)
        self.progress_bar.pack(side="left", fill="x", expand=True)
        self.frame_label = ttk.Label(progress_frame, text="0 / 0", width=12)
        self.frame_label.pack(side="right")

        # 播放控制
        control_frame = ttk.Frame(right_frame)
        control_frame.pack(fill="x", padx=4, pady=4)

        self.play_btn = ttk.Button(control_frame, text="▶ 播放", command=self._toggle_play, width=10)
        self.play_btn.pack(side="left", padx=2)
        ttk.Button(control_frame, text="⏭ 下一帧", command=self._step_forward, width=10).pack(side="left", padx=2)
        ttk.Button(control_frame, text="⏹ 停止", command=self._stop_processing, width=10).pack(side="left", padx=2)
        ttk.Button(control_frame, text="⏮ 回到开头", command=self._rewind, width=10).pack(side="left", padx=2)

        # 底部状态栏
        self.status_bar = ttk.Label(self.root, text="就绪", relief="sunken", anchor="w")
        self.status_bar.pack(fill="x", side="bottom")

        # 鼠标滚轮滚动
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

    def _add_slider(self, parent, label_text, param_name, min_val, max_val, default, step):
        """添加参数滑块到 UI"""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=2)
        ttk.Label(frame, text=label_text, width=12, justify="left").pack(side="left")
        var = tk.DoubleVar(value=default) if isinstance(default, float) else tk.IntVar(value=default)
        setattr(self, f"{param_name}_var", var)
        scale = ttk.Scale(frame, from_=min_val, to=max_val, variable=var, length=80,
                          command=lambda v, name=param_name: self._on_slider_change(name))
        scale.pack(side="left", padx=4)
        val_label = ttk.Label(frame, text=f"{default:.2f}" if isinstance(default, float) else str(default), width=6)
        val_label.pack(side="right")
        setattr(self, f"{param_name}_label", val_label)

    # ── 模型加载 ──────────────────────────────────────────────────────
    def _try_load_model(self):
        """尝试从默认路径加载模型"""
        path = Path(self.model_path)
        if path.exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(path))
                self.model_loaded = True
                self.model_label.config(text=f"✅ {path.name}", foreground="green")
                self._set_status(f"模型已加载: {path.name}")
            except Exception as e:
                self.model_label.config(text=f"❌ 加载失败: {e}", foreground="red")
                self._set_status(f"模型加载失败: {e}")
        else:
            self.model_label.config(text="⚠ 权重文件不存在，请选择模型文件", foreground="orange")
            self._set_status(f"权重文件不存在: {path}")

    def _select_model(self):
        """让用户选择模型文件"""
        path = filedialog.askopenfilename(
            title="选择模型权重文件",
            filetypes=[("PyTorch模型", "*.pt"), ("ONNX模型", "*.onnx"), ("所有文件", "*.*")]
        )
        if not path:
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(path)
            self.model_loaded = True
            self.model_path = path
            self.model_label.config(text=f"✅ {Path(path).name}", foreground="green")
            self._set_status(f"模型已加载: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("模型加载失败", str(e))
            self._set_status(f"模型加载失败: {e}")

    # ── 视频导入 ──────────────────────────────────────────────────────
    def _import_video(self):
        """导入视频文件"""
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"), ("所有文件", "*.*")]
        )
        if not path:
            return

        # 关闭旧视频
        if self.cap:
            self.cap.release()
            self.cap = None

        self._stop_processing()

        self.video_path = path

        # 处理中文/空格路径：复制到临时英文路径
        if has_non_ascii(path) or " " in path:
            TEMP_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
            ext = Path(path).suffix
            temp_name = f"{uuid.uuid4().hex}{ext}"
            self.temp_video_path = str(TEMP_VIDEO_DIR / temp_name)
            self._set_status("复制视频到临时路径（规避中文路径）...")
            try:
                shutil.copy2(path, self.temp_video_path)
            except Exception as e:
                messagebox.showerror("视频导入失败", f"复制文件失败:\n{e}")
                return
        else:
            self.temp_video_path = path

        # 打开视频
        self.cap = cv2.VideoCapture(self.temp_video_path)
        if not self.cap.isOpened():
            messagebox.showerror("错误", "无法打开视频文件")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_idx = 0

        # 重置跟踪
        self.tracker_initialized = False
        self.active_ids.clear()
        self.all_ids.clear()
        self.id_first_frame.clear()
        self.id_last_frame.clear()

        # 更新 UI
        fname = Path(path).name
        self.video_label.config(
            text=f"📹 {fname}\n   {self.total_frames}帧 | {self.fps:.1f}FPS | {self.orig_w}x{self.orig_h}",
            foreground="black"
        )
        self.progress_bar.config(to=self.total_frames)
        self.progress_var.set(0)
        self.frame_label.config(text=f"0 / {self.total_frames}")
        self._set_status(f"视频已导入: {fname}")

        # 显示第一帧
        ret, frame = self.cap.read()
        if ret:
            self._display_frame(frame)
            self.current_frame_idx = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── 视频处理（工作线程）────────────────────────────────────────
    def _process_video(self):
        """工作线程：逐帧读取 → 推理 → 跟踪 → 绘制 → 存储"""
        try:
            while not self.stop_event.is_set():
                # 暂停检查
                self.pause_event.wait()
                if self.stop_event.is_set():
                    break

                # 读取帧
                if self.is_saving:
                    ret, frame = self.cap.read()
                else:
                    # 非保存模式：暂停时不读帧
                    if not self.is_playing:
                        time.sleep(0.01)
                        continue
                    ret, frame = self.cap.read()
                if not ret:
                    # 视频结束
                    if self.is_saving:
                        self._finish_save()
                    else:
                        self.is_playing = False
                        self.root.after(0, lambda: self.play_btn.config(text="▶ 播放"))
                        self._set_status("视频播放完毕（停在最后一帧）")
                    break

                self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

                # 推理 + 跟踪
                annotated = self._infer_and_track(frame)

                # 保存写入
                if self.is_saving and self.writer is not None:
                    self.writer.write(annotated)

                # 存储最新帧供主线程显示
                with self.frame_lock:
                    self.latest_frame = annotated.copy()

                # 更新进度（主线程）
                progress = (self.current_frame_idx + 1) / max(1, self.total_frames) * 100
                self.root.after(0, lambda p=progress, idx=self.current_frame_idx:
                                self._update_progress(p, idx))

        except Exception as e:
            err_msg = f"处理错误: {e}"
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self._set_status(err_msg))
            self.root.after(0, lambda: messagebox.showerror("处理错误", err_msg))
        finally:
            # 始终释放 writer，防止资源泄漏
            if self.writer:
                self._finish_save()

    def _infer_and_track(self, frame: np.ndarray) -> np.ndarray:
        """
        对单帧执行推理 + 跟踪，返回标注后的帧。

        逻辑步骤：
        1. 如果跟踪器未初始化，生成 YAML 并首次调用 model.track(persist=True)
        2. 后续帧调用 model.track(persist=True) 复用跟踪器
        3. 从结果中提取检测框 + 跟踪ID，绘制到帧上
        4. 更新统计信息
        """
        if not self.model_loaded or self.model is None:
            return frame

        # 收集当前参数
        params = self._collect_params()

        try:
            if not self.tracker_initialized:
                # 首次调用：生成 YAML，初始化跟踪器
                self.tracker_yaml_path = generate_tracker_yaml(params, self.tracker_type)
                self._set_status("初始化跟踪器...")

                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker_yaml_path,
                    conf=self.conf_var.get(),
                    iou=self.iou_var.get(),
                    imgsz=self.imgsz_var.get(),
                    verbose=False,
                )
                self.tracker_initialized = True
                self._set_status("跟踪器已初始化，开始推理...")
            else:
                # 后续帧：复用跟踪器，实时参数已通过 _update_tracker_params 更新
                results = self.model.track(
                    source=frame,
                    persist=True,
                    conf=self.conf_var.get(),
                    iou=self.iou_var.get(),
                    imgsz=self.imgsz_var.get(),
                    verbose=False,
                )
        except Exception as e:
            self._set_status(f"推理错误: {e}")
            return frame

        # 绘制结果
        annotated = frame.copy()
        current_ids = set()

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                # 获取跟踪 ID
                if boxes.id is not None:
                    ids = boxes.id.cpu().numpy().astype(int)
                else:
                    ids = [-1] * len(boxes)

                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = xyxy[i].astype(int)
                    tid = int(ids[i])
                    conf = float(confs[i])

                    if tid < 0:
                        continue

                    color = get_track_color(tid)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                    # ID 标签
                    label = f"ID:{tid} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
                    cv2.putText(annotated, label, (x1 + 1, y1 - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # 更新统计
                    current_ids.add(tid)
                    self.all_ids.add(tid)
                    if tid not in self.id_first_frame:
                        self.id_first_frame[tid] = self.current_frame_idx
                    self.id_last_frame[tid] = self.current_frame_idx

        self.active_ids = current_ids
        self._update_stats()

        return annotated

    # ── 实时参数更新 ──────────────────────────────────────────────────
    def _collect_params(self) -> dict:
        """从 UI 控件收集当前跟踪参数"""
        return {
            "track_buffer": int(self.track_buffer_var.get()),
            "match_thresh": round(self.match_thresh_var.get(), 4),
            "track_high_thresh": round(self.track_high_thresh_var.get(), 4),
            "track_low_thresh": round(self.track_low_thresh_var.get(), 4),
            "new_track_thresh": round(self.new_track_thresh_var.get(), 4),
            "lost_match_thr": round(self.lost_match_thr_var.get(), 4),
        }

    def _update_tracker_params(self):
        """
        实时修改跟踪器参数（下一帧生效）。

        原理：
        - model.track(persist=True) 首次调用后，predictor.trackers[0] 持久存在
        - tracker.update() 每帧读取 self.args.* 和 self.max_frames_lost
        - 直接修改这些属性，下一帧立即使用新值
        """
        if not self.tracker_initialized or self.model is None:
            return

        try:
            predictor = self.model.predictor
            if hasattr(predictor, "trackers") and len(predictor.trackers) > 0:
                tracker = predictor.trackers[0]
                params = self._collect_params()

                # 修改 max_frames_lost（对应 track_buffer）
                tracker.max_frames_lost = params["track_buffer"]

                # 修改 args 属性（match_thresh, track_high_thresh 等）
                tracker.args.match_thresh = params["match_thresh"]
                tracker.args.track_high_thresh = params["track_high_thresh"]
                tracker.args.track_low_thresh = params["track_low_thresh"]
                tracker.args.new_track_thresh = params["new_track_thresh"]

                # TrackTrack 的 lost_match_thr
                if self.tracker_type == "tracktrack":
                    if hasattr(tracker.args, "lost_match_thr"):
                        tracker.args.lost_match_thr = params["lost_match_thr"]
        except Exception as e:
            self._set_status(f"参数更新警告: {e}")

    def _on_slider_change(self, param_name: str):
        """滑块值变化时更新显示并实时修改跟踪器参数"""
        var = getattr(self, f"{param_name}_var")
        label = getattr(self, f"{param_name}_label", None)
        val = var.get()
        if label:
            if param_name == "track_buffer":
                label.config(text=str(int(val)))
            else:
                label.config(text=f"{val:.2f}")
        # 实时更新跟踪器
        self._update_tracker_params()

    def _on_preset_change(self):
        """宽容度预设变化时设置滑块值"""
        preset_name = self.preset_var.get()
        if preset_name not in TOLERANCE_PRESETS:
            return
        preset = TOLERANCE_PRESETS[preset_name]
        for key, val in preset.items():
            var = getattr(self, f"{key}_var", None)
            label = getattr(self, f"{key}_label", None)
            if var is not None:
                var.set(val)
                if label:
                    if key == "track_buffer":
                        label.config(text=str(int(val)))
                    else:
                        label.config(text=f"{val:.2f}")
        self._update_tracker_params()
        self._set_status(f"已应用预设: {preset_name}")

    def _on_tracker_type_change(self):
        """跟踪器类型变化时切换 UI"""
        new_type = self.tracker_type_var.get()
        if new_type == self.tracker_type:
            return
        self.tracker_type = new_type

        # 显示/隐藏 lost_match_thr 控件（放在重置按钮之前）
        if new_type == "tracktrack":
            self.lost_match_frame.pack(fill="x", pady=2, before=self.reset_track_btn)
        else:
            self.lost_match_frame.pack_forget()

        # 需要重置跟踪
        self._reset_tracking()
        self._set_status(f"已切换跟踪器: {new_type}（需重置跟踪生效）")

    def _on_imgsz_change(self):
        """推理分辨率变化"""
        self._set_status(f"推理分辨率: {self.imgsz_var.get()}")

    # ── 跟踪重置 ──────────────────────────────────────────────────────
    def _reset_tracking(self):
        """重置跟踪器状态（从头开始跟踪）"""
        was_playing = self.is_playing
        self._stop_processing()

        # 重置跟踪器
        self.tracker_initialized = False
        self.active_ids.clear()
        self.all_ids.clear()
        self.id_first_frame.clear()
        self.id_last_frame.clear()

        # 回到视频开头
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_idx = 0
            ret, frame = self.cap.read()
            if ret:
                self._display_frame(frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.progress_var.set(0)
        self.frame_label.config(text=f"0 / {self.total_frames}")
        self._update_stats()
        self._set_status("跟踪已重置，可重新播放")

    # ── 播放控制 ──────────────────────────────────────────────────────
    def _toggle_play(self):
        """播放/暂停切换"""
        if not self.cap:
            messagebox.showwarning("提示", "请先导入视频")
            return

        if not self.model_loaded:
            messagebox.showwarning("提示", "请先加载模型")
            return

        if self.is_playing:
            # 暂停
            self.is_playing = False
            self.pause_event.clear()
            self.play_btn.config(text="▶ 播放")
            self._set_status("已暂停")
        else:
            # 播放
            self.is_playing = True
            self.pause_event.set()
            self.stop_event.clear()
            self.play_btn.config(text="⏸ 暂停")

            # 启动工作线程（如果未运行）
            if self.worker_thread is None or not self.worker_thread.is_alive():
                self.worker_thread = threading.Thread(target=self._process_video, daemon=True)
                self.worker_thread.start()

            self._set_status("播放中...")

    def _stop_processing(self):
        """停止处理"""
        self.is_playing = False
        self.is_saving = False
        self.pause_event.set()
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=3)
        self.worker_thread = None
        self.stop_event.clear()
        self.root.after(0, lambda: self.play_btn.config(text="▶ 播放"))

    def _step_forward(self):
        """前进一帧"""
        if not self.cap or not self.model_loaded:
            return

        self.is_playing = False
        self.pause_event.clear()
        self.play_btn.config(text="▶ 播放")

        ret, frame = self.cap.read()
        if not ret:
            self._set_status("已到视频末尾")
            return

        self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        annotated = self._infer_and_track(frame)
        with self.frame_lock:
            self.latest_frame = annotated.copy()
        self._display_frame(annotated)
        self._update_progress(
            (self.current_frame_idx + 1) / max(1, self.total_frames) * 100,
            self.current_frame_idx
        )

    def _rewind(self):
        """回到视频开头"""
        if not self.cap:
            return
        self._stop_processing()
        self.tracker_initialized = False
        self.active_ids.clear()
        self.all_ids.clear()
        self.id_first_frame.clear()
        self.id_last_frame.clear()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if ret:
            self._display_frame(frame)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame_idx = 0
        self.progress_var.set(0)
        self.frame_label.config(text=f"0 / {self.total_frames}")
        self._update_stats()
        self._set_status("已回到开头")

    def _on_progress_seek(self, val):
        """进度条拖动跳转（自动暂停播放）"""
        if not self.cap:
            return
        # 暂停播放
        if self.is_playing:
            self.is_playing = False
            self.pause_event.clear()
            self.play_btn.config(text="▶ 播放")
        target = int(float(val))
        if target >= self.total_frames:
            target = self.total_frames - 1
        if target < 0:
            target = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = self.cap.read()
        if ret:
            self._display_frame(frame)
            self.current_frame_idx = target
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target + 1)
        self.frame_label.config(text=f"{target} / {self.total_frames}")

    # ── 帧显示（主线程轮询）──────────────────────────────────────────
    def _poll_frame(self):
        """主线程定期轮询最新帧并更新显示"""
        with self.frame_lock:
            if self.latest_frame is not None:
                self._display_frame(self.latest_frame)
        self.root.after(33, self._poll_frame)  # ~30fps 更新

    def _display_frame(self, frame):
        """在画布上显示帧（等比缩放）"""
        if frame is None:
            return
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        h, w = frame.shape[:2]
        scale = min(canvas_w / w, canvas_h / h, 1.0)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        small = cv2.resize(frame, (nw, nh))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image((canvas_w - nw) // 2, (canvas_h - nh) // 2, anchor="nw", image=photo)
        self.canvas.image = photo  # 防止 GC

    def _on_canvas_resize(self):
        """画布尺寸变化时重绘最新帧"""
        with self.frame_lock:
            if self.latest_frame is not None:
                self._display_frame(self.latest_frame)

    # ── 统计更新 ──────────────────────────────────────────────────────
    def _update_stats(self):
        """更新跟踪统计信息"""
        max_lifespan = 0
        for tid in self.all_ids:
            lifespan = self.id_last_frame.get(tid, 0) - self.id_first_frame.get(tid, 0)
            if lifespan > max_lifespan:
                max_lifespan = lifespan
        text = f"当前ID: {len(self.active_ids)}\n总ID: {len(self.all_ids)}\n最长寿命: {max_lifespan}帧"
        self.root.after(0, lambda: self.stats_label.config(text=text))

    def _update_progress(self, progress, frame_idx):
        """更新进度条"""
        self.progress_var.set(frame_idx + 1)
        self.frame_label.config(text=f"{frame_idx + 1} / {self.total_frames}")

    # ── 视频保存 ──────────────────────────────────────────────────────
    def _save_video(self):
        """保存带跟踪标注的视频"""
        if not self.cap or not self.model_loaded:
            messagebox.showwarning("提示", "请先导入视频并加载模型")
            return

        # 选择保存路径
        default_name = f"{Path(self.video_path).stem}_tracked.avi"
        save_path = filedialog.asksaveasfilename(
            title="保存跟踪视频",
            defaultextension=".avi",
            initialfile=default_name,
            filetypes=[("AVI视频", "*.avi"), ("MP4视频", "*.mp4"), ("所有文件", "*.*")]
        )
        if not save_path:
            return

        # 停止当前处理
        self._stop_processing()

        # 重置到开头
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.tracker_initialized = False
        self.active_ids.clear()
        self.all_ids.clear()
        self.id_first_frame.clear()
        self.id_last_frame.clear()

        # 创建 VideoWriter（根据文件扩展名选择 fourcc）
        ext = Path(save_path).suffix.lower()
        if ext == ".avi":
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
        elif ext == ".mp4":
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(save_path, fourcc, self.fps, (self.orig_w, self.orig_h))
        if not self.writer.isOpened():
            messagebox.showerror("错误", f"无法创建视频写入器 (fourcc={fourcc}, ext={ext})")
            return

        # 启动保存模式
        self.is_saving = True
        self.stop_event.clear()
        self.pause_event.set()
        self.worker_thread = threading.Thread(target=self._process_video, daemon=True)
        self.worker_thread.start()
        self._set_status(f"正在保存到: {Path(save_path).name}...")

    def _finish_save(self):
        """完成视频保存"""
        if self.writer:
            self.writer.release()
            self.writer = None
        self.is_saving = False
        self.root.after(0, lambda: self._set_status("视频保存完成"))

    # ── 缓存清理 ──────────────────────────────────────────────────────
    def _clear_cache(self):
        """清理临时文件"""
        cleaned = 0
        if TEMP_DIR.exists():
            try:
                shutil.rmtree(TEMP_DIR)
                cleaned += 1
            except Exception:
                pass
        self.temp_video_path = None
        self.tracker_yaml_path = None
        self._set_status(f"已清理临时缓存" if cleaned else "无缓存需清理")
        messagebox.showinfo("清理完成", f"已清理临时文件目录:\n{TEMP_DIR}")

    # ── 辅助 ──────────────────────────────────────────────────────────
    def _set_status(self, text):
        """更新状态栏"""
        self.root.after(0, lambda: self.status_bar.config(text=text))

    def on_close(self):
        """窗口关闭时的清理"""
        self._stop_processing()
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
        self.root.destroy()


# ── 入口 ──────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    app = VideoTrackerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
