# -*- coding: utf-8 -*-
"""
摄像头连续录制工具 v2（多线程无卡顿版）
================================================================
v2 性能优化：
  1. 独立帧采集线程（生产者）：持续从摄像头读取帧，存入有界最新帧队列
  2. 预览和录制（消费者）都从队列取帧，不再直接调用 cap.read()
  3. 队列满时自动丢弃旧帧，保证预览和录制始终拿到最新帧，不积压不延迟
  4. 录制时预览自动降频（15fps），减少主线程负担
  5. UI进度更新限流（每0.2秒一次），避免每帧调度导致卡顿
  6. 彻底解决大分辨率（1920x1080）下预览与录制竞争导致的卡顿丢帧

功能：检测可用摄像头 → 实时预览画面 → 设定录制时长/组数/保存地址 → 连续录制多段视频
新增：HY-500B 适配模块（一键应用 MSMF 后端 + 分辨率选择 + 参数可调）
依赖：opencv-python, pillow, numpy
================================================================
"""
import sys
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageTk
import cv2

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

APP_TITLE = "摄像头连续录制工具"

# HY-500B 适配配置
HY500B_CONFIG = {
    "backend": cv2.CAP_MSMF,
    "default_resolution": (1920, 1080),
    "resolutions": [
        (2048, 1536, "2048x1536 (3MP)", 29.9),
        (1920, 1080, "1920x1080 (1080p)", 30.2),
        (1600, 1200, "1600x1200 (UXGA)", 30.0),
        (1280, 960, "1280x960 (SXGA)", 29.8),
        (1280, 720, "1280x720 (720p)", 30.2),
        (800, 600, "800x600 (SVGA)", 29.9),
        (640, 480, "640x480 (VGA)", 30.2),
    ],
    "params": {
        "brightness": {"label": "亮度", "prop": cv2.CAP_PROP_BRIGHTNESS, "min": 0, "max": 255, "default": 0},
        "contrast":   {"label": "对比度", "prop": cv2.CAP_PROP_CONTRAST,   "min": 0, "max": 255, "default": 32},
        "saturation": {"label": "饱和度", "prop": cv2.CAP_PROP_SATURATION, "min": 0, "max": 255, "default": 60},
        "gain":       {"label": "增益", "prop": cv2.CAP_PROP_GAIN,       "min": 0, "max": 255, "default": 0},
    },
}

# 帧队列最大容量：2 帧缓冲，满时丢弃旧帧保证实时性
FRAME_QUEUE_MAXSIZE = 2
# 预览正常帧率间隔（毫秒）
PREVIEW_INTERVAL_NORMAL = 33   # ~30fps
# 录制时预览降频间隔（毫秒）
PREVIEW_INTERVAL_RECORDING = 66  # ~15fps，减少主线程负担


class CameraRecorderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("960x760")
        self.root.minsize(760, 600)

        # 摄像头状态
        self.cap: cv2.VideoCapture | None = None
        self.camera_index: int | None = None
        self.available_cameras: list[tuple] = []
        self.preview_running = False
        self.photo: ImageTk.PhotoImage | None = None

        # ── v2 多线程帧采集 ──
        self.frame_queue: queue.Queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
        self.capture_thread: threading.Thread | None = None
        self.capture_running = False  # 采集线程运行标志
        self._capture_lock = threading.Lock()  # 保护 cap 访问

        # 录制状态
        self.recording = False
        self.record_thread: threading.Thread | None = None
        self.stop_flag = False

        # HY-500B 适配模式
        self.hy500b_mode = False
        self.param_vars: dict[str, tk.IntVar] = {}
        self.param_labels: dict[str, ttk.Label] = {}
        self._param_sync_lock = False
        self.current_resolution = HY500B_CONFIG["default_resolution"]
        # 参数防抖：滑块拖动时不立即调用 cap.set()，停止拖动后批量应用
        self._pending_params: dict[str, int] = {}
        self._param_apply_after_id: str | None = None
        self._resolution_apply_after_id: str | None = None
        # 上一个成功的分辨率，用于高分辨率切换失败时回退
        self._last_resolution = HY500B_CONFIG["default_resolution"]
        self._resolution_changing = False  # 分辨率切换中标志，防止重复切换

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI 构建 ──
    def _build_ui(self) -> None:
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=6)

        ttk.Button(top, text="检测摄像头", command=self._on_detect).pack(side="left", padx=2)
        ttk.Label(top, text="摄像头:").pack(side="left", padx=(8, 2))
        self.cam_var = tk.StringVar(value="未检测")
        self.cam_combo = ttk.Combobox(top, textvariable=self.cam_var, width=28, state="readonly")
        self.cam_combo.pack(side="left", padx=2)
        self.cam_combo.bind("<<ComboboxSelected>>", self._on_camera_select)

        self.cam_info_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.cam_info_var, foreground="#0066cc").pack(side="left", padx=12)

        self.hy500b_btn = ttk.Button(
            top, text="⚙ 应用 HY-500B 适配", command=self._toggle_hy500b)
        self.hy500b_btn.pack(side="right", padx=2)

        # HY-500B 参数调节面板
        self.param_frame = ttk.LabelFrame(self.root, text="HY-500B 参数调节（实时生效）", padding=6)

        res_row = ttk.Frame(self.param_frame)
        res_row.pack(fill="x", pady=(0, 4))
        ttk.Label(res_row, text="分辨率:", font=("Microsoft YaHei", 10, "bold")).pack(side="left", padx=2)
        self.resolution_var = tk.StringVar()
        self.resolution_combo = ttk.Combobox(
            res_row, textvariable=self.resolution_var, width=28, state="readonly",
            values=[r[2] for r in HY500B_CONFIG["resolutions"]])
        self.resolution_combo.pack(side="left", padx=4)
        self.resolution_combo.bind("<<ComboboxSelected>>", self._on_resolution_change)
        self.resolution_fps_var = tk.StringVar(value="")
        ttk.Label(res_row, textvariable=self.resolution_fps_var,
                  foreground="#008800").pack(side="left", padx=8)
        default_label = f"{HY500B_CONFIG['default_resolution'][0]}x{HY500B_CONFIG['default_resolution'][1]}"
        for i, r in enumerate(HY500B_CONFIG["resolutions"]):
            if r[0] == HY500B_CONFIG["default_resolution"][0] and r[1] == HY500B_CONFIG["default_resolution"][1]:
                self.resolution_combo.current(i)
                self.resolution_fps_var.set(f"实测约 {r[3]:.0f} fps")
                break

        param_row = ttk.Frame(self.param_frame)
        param_row.pack(fill="x")

        for i, (key, cfg) in enumerate(HY500B_CONFIG["params"].items()):
            col_frame = ttk.Frame(param_row)
            col_frame.grid(row=0, column=i, padx=8, pady=2, sticky="w")

            var = tk.IntVar(value=cfg["default"])
            self.param_vars[key] = var

            lbl = ttk.Label(col_frame, text=f"{cfg['label']}: {cfg['default']}")
            lbl.pack(side="top", anchor="w")
            self.param_labels[key] = lbl

            scale = ttk.Scale(
                col_frame, from_=cfg["min"], to=cfg["max"],
                orient="horizontal", length=140,
                command=lambda val, k=key: self._on_param_change(k, val))
            scale.set(cfg["default"])
            scale.pack(side="top", fill="x")

        btn_row = ttk.Frame(self.param_frame)
        btn_row.pack(fill="x", pady=(4, 0))
        ttk.Button(btn_row, text="重置为默认值", width=14,
                   command=self._reset_params).pack(side="left", padx=2)
        ttk.Button(btn_row, text="读取当前值", width=14,
                   command=self._read_params_from_camera).pack(side="left", padx=2)
        self.param_status_var = tk.StringVar(value="")
        ttk.Label(btn_row, textvariable=self.param_status_var,
                  foreground="#008800").pack(side="left", padx=10)

        # 中部：预览画布
        mid = ttk.Frame(self.root)
        mid.pack(fill="both", expand=True, padx=8, pady=4)

        self.canvas = tk.Canvas(mid, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # 预览缩放控制条
        zoom_bar = ttk.Frame(self.root)
        zoom_bar.pack(fill="x", padx=8, pady=(0, 2))

        self.zoom_fit_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(zoom_bar, text="适应窗口", variable=self.zoom_fit_var,
                        command=self._on_zoom_fit_toggle).pack(side="left", padx=2)

        ttk.Label(zoom_bar, text="缩放:").pack(side="left", padx=(8, 2))
        self.zoom_var = tk.IntVar(value=100)
        self.zoom_scale = ttk.Scale(
            zoom_bar, from_=10, to=300, orient="horizontal", length=200,
            variable=self.zoom_var, command=self._on_zoom_change, state="disabled")
        self.zoom_scale.pack(side="left", padx=2)
        self.zoom_label = ttk.Label(zoom_bar, text="100%", width=6, foreground="#0066cc")
        self.zoom_label.pack(side="left", padx=2)

        ttk.Button(zoom_bar, text="−", width=3, command=lambda: self._zoom_step(-10)).pack(side="left", padx=1)
        ttk.Button(zoom_bar, text="+", width=3, command=lambda: self._zoom_step(10)).pack(side="left", padx=1)
        ttk.Button(zoom_bar, text="1:1", width=4, command=self._zoom_actual).pack(side="left", padx=1)

        # 底部：录制参数
        bot = ttk.Frame(self.root)
        bot.pack(fill="x", padx=8, pady=6)

        ttk.Label(bot, text="录制时长(秒):").pack(side="left", padx=2)
        self.duration_var = tk.IntVar(value=30)
        ttk.Spinbox(bot, from_=1, to=3600, width=6, textvariable=self.duration_var).pack(side="left", padx=2)

        ttk.Label(bot, text="组数:").pack(side="left", padx=(8, 2))
        self.groups_var = tk.IntVar(value=3)
        ttk.Spinbox(bot, from_=1, to=999, width=5, textvariable=self.groups_var).pack(side="left", padx=2)

        ttk.Label(bot, text="保存到:").pack(side="left", padx=(8, 2))
        self.save_dir_var = tk.StringVar(value=str(Path.home() / "Videos"))
        ttk.Entry(bot, textvariable=self.save_dir_var, width=22).pack(side="left", padx=2)
        ttk.Button(bot, text="浏览", width=5, command=self._on_browse).pack(side="left", padx=2)

        ttk.Separator(bot, orient="vertical").pack(side="left", fill="y", padx=8)

        self.record_btn = ttk.Button(bot, text="● 开始录制", command=self._on_toggle_record)
        self.record_btn.pack(side="right", padx=2)

        # 状态栏
        status = ttk.Frame(self.root)
        status.pack(fill="x", padx=8, pady=(0, 4))
        self.status_var = tk.StringVar(value="请先点击「检测摄像头」")
        ttk.Label(status, textvariable=self.status_var, foreground="#333").pack(side="left")
        self.progress_var = tk.StringVar(value="")
        ttk.Label(status, textvariable=self.progress_var,
                  foreground="#cc3333", font=("Microsoft YaHei", 10, "bold")).pack(side="right")

    # ── v2 帧采集线程（生产者） ──
    def _start_capture_thread(self) -> None:
        """启动独立帧采集线程，持续读取摄像头帧并存入队列"""
        if self.capture_running:
            return
        self.capture_running = True
        # 清空队列，避免旧帧残留
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _stop_capture_thread(self) -> None:
        """停止帧采集线程"""
        self.capture_running = False
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        self.capture_thread = None
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

    def _capture_loop(self) -> None:
        """帧采集线程主循环：持续读取，队列满时丢弃旧帧，保证最新帧可用"""
        while self.capture_running:
            with self._capture_lock:
                if self.cap is None:
                    time.sleep(0.01)
                    continue
                ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
            # 队列满时丢弃最旧帧，保证队列中始终是最新帧
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

    def _get_latest_frame(self):
        """从队列获取最新帧（非阻塞，无帧返回 None）"""
        frame = None
        # 持续取直到队列空，最后一帧就是最新帧
        while True:
            try:
                frame = self.frame_queue.get_nowait()
            except queue.Empty:
                break
        return frame

    # ── HY-500B 适配模块 ──
    def _toggle_hy500b(self) -> None:
        if self.recording:
            messagebox.showinfo(APP_TITLE, "录制中无法切换适配模式，请先停止录制")
            return

        if self.hy500b_mode:
            self.hy500b_mode = False
            self.hy500b_btn.config(text="⚙ 应用 HY-500B 适配")
            self.param_frame.pack_forget()
            self.status_var.set("已切换为默认模式")
            if self.available_cameras:
                self._on_detect()
        else:
            self.hy500b_mode = True
            self.hy500b_btn.config(text="✕ 取消 HY-500B 适配")
            self.param_frame.pack(fill="x", padx=8, pady=(0, 4), before=self.root.children.get('!frame2'))
            self.status_var.set("HY-500B 适配已启用：MSMF后端 + 1920x1080 + 参数可调")
            self._on_detect()

    def _get_backend(self):
        return HY500B_CONFIG["backend"] if self.hy500b_mode else None

    def _on_param_change(self, key: str, val: str) -> None:
        """参数滑块变化：实时应用到摄像头（不持锁、不防抖，MSMF驱动内部有同步）"""
        if self._param_sync_lock:
            return
        try:
            value = int(float(val))
        except (ValueError, TypeError):
            return

        cfg = HY500B_CONFIG["params"][key]
        self.param_vars[key].set(value)
        self.param_labels[key].config(text=f"{cfg['label']}: {value}")

        if self.cap is None:
            # 分辨率切换中 cap 暂不可用，记录到待应用队列，切换完成后补应用
            self._pending_params[key] = value
            return

        # 实时直接设置（不持有采集锁，MSMF驱动内部有同步）
        ok = self.cap.set(cfg["prop"], value)
        if not ok:
            # 设置失败：记录到待应用队列，稍后重试
            self._pending_params[key] = value
            if self._param_apply_after_id is None:
                self._param_apply_after_id = self.root.after(500, self._apply_pending_params)
        else:
            # 设置成功：清除该参数的待应用记录
            self._pending_params.pop(key, None)
            self.param_status_var.set(f"已应用 {cfg['label']}={value}")
            self.root.after(1000, lambda: self.param_status_var.set(""))

    def _apply_pending_params(self) -> None:
        """补应用待设置参数（分辨率切换完成后或设置失败后重试）"""
        self._param_apply_after_id = None
        if not self._pending_params or self.cap is None:
            if self._pending_params and self.cap is None:
                self._param_apply_after_id = self.root.after(300, self._apply_pending_params)
            return

        failed = {}
        for key, value in list(self._pending_params.items()):
            cfg = HY500B_CONFIG["params"][key]
            ok = self.cap.set(cfg["prop"], value)
            if ok:
                self._pending_params.pop(key, None)
            else:
                failed[key] = value

        if failed:
            self._param_apply_after_id = self.root.after(500, self._apply_pending_params)
        else:
            self.param_status_var.set("参数已同步")
            self.root.after(1000, lambda: self.param_status_var.set(""))

    def _on_resolution_change(self, _event=None) -> None:
        """分辨率下拉变化：防抖延迟切换，避免MSMF后端cap.set()慢导致连续切换卡死"""
        if self._param_sync_lock or self._resolution_changing:
            return
        idx = self.resolution_combo.current()
        if idx < 0:
            return
        w, h, label, fps = HY500B_CONFIG["resolutions"][idx]

        # 如果和当前分辨率相同，不处理
        if self.current_resolution[0] == w and self.current_resolution[1] == h:
            return

        self.current_resolution = (w, h)
        self.resolution_fps_var.set(f"实测约 {fps:.0f} fps")

        if self.cap is None:
            self.param_status_var.set(f"已选择 {label}，开启摄像头后生效")
            self.root.after(2000, lambda: self.param_status_var.set(""))
            return

        if self.recording:
            messagebox.showinfo(APP_TITLE, "录制中无法切换分辨率，请先停止录制")
            for i, r in enumerate(HY500B_CONFIG["resolutions"]):
                if r[0] == self.current_resolution[0] and r[1] == self.current_resolution[1]:
                    self._param_sync_lock = True
                    self.resolution_combo.current(i)
                    self._param_sync_lock = False
                    break
            return

        # 防抖：取消之前的延迟切换，300ms内无新选择才真正切换
        if self._resolution_apply_after_id is not None:
            self.root.after_cancel(self._resolution_apply_after_id)
        self._resolution_apply_after_id = self.root.after(
            300, lambda: self._apply_resolution_change(w, h, label, fps))

    def _apply_resolution_change(self, w: int, h: int, label: str, fps: float) -> None:
        """启动后台线程执行分辨率切换（不阻塞主线程，避免高分辨率下cap.set()卡死UI）"""
        self._resolution_apply_after_id = None
        if self.cap is None or self._resolution_changing:
            return
        if self.camera_index is None:
            return

        self._resolution_changing = True
        # 禁用所有摄像头相关控件，防止切换中重复操作
        self.resolution_combo.config(state="disabled")
        self.cam_combo.config(state="disabled")
        self.record_btn.config(state="disabled")
        self.param_status_var.set(f"正在切换到 {label}，请稍候...")

        # 启动后台线程执行实际切换
        t = threading.Thread(
            target=self._apply_resolution_worker,
            args=(w, h, label, self.camera_index),
            daemon=True)
        t.start()

    def _apply_resolution_worker(self, w: int, h: int, label: str, cam_idx: int) -> None:
        """后台线程：释放旧cap → 重新打开 → 设置分辨率 → 验证读取 → 成功则替换self.cap，失败则回退"""
        backend = self._get_backend()
        old_w, old_h = self._last_resolution

        # 1. 停止预览和采集线程
        self.preview_running = False
        self.capture_running = False
        # 短暂等待采集线程退出（不强制join，避免cap.read()阻塞导致死等）
        time.sleep(0.3)

        # 2. 释放旧 cap
        old_cap = self.cap
        self.cap = None
        try:
            if old_cap is not None:
                old_cap.release()
        except Exception:
            pass
        time.sleep(0.2)  # 等待驱动释放设备

        # 3. 重新打开摄像头并设置新分辨率
        try:
            if backend is not None:
                new_cap = cv2.VideoCapture(cam_idx, backend)
            else:
                new_cap = cv2.VideoCapture(cam_idx)

            if not new_cap.isOpened():
                raise RuntimeError("无法重新打开摄像头")

            new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

            actual_w = int(new_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(new_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = new_cap.get(cv2.CAP_PROP_FPS)
            if actual_fps <= 0 or actual_fps != actual_fps:
                actual_fps = 25.0

            # 4. 验证：连续读取5帧，确认高分辨率下能正常取帧
            verify_ok = False
            for _ in range(5):
                ret, test_frame = new_cap.read()
                if ret and test_frame is not None and test_frame.shape[0] > 0:
                    verify_ok = True
                    break
                time.sleep(0.05)

            if not verify_ok:
                raise RuntimeError(f"设置 {label} 后无法读取帧")

            # 5. 成功：应用参数，替换 self.cap
            if self.hy500b_mode:
                for key, cfg in HY500B_CONFIG["params"].items():
                    new_cap.set(cfg["prop"], self.param_vars[key].get())

            with self._capture_lock:
                self.cap = new_cap
            self._last_resolution = (actual_w, actual_h)

            # 6. 通知主线程切换成功
            self.root.after(0, lambda: self._on_resolution_change_done(
                success=True, actual_w=actual_w, actual_h=actual_h,
                actual_fps=actual_fps, label=label))

        except Exception as e:
            # 切换失败：回退到上一个分辨率
            safe_print(f"[分辨率切换失败] {label}: {e}，回退到 {old_w}x{old_h}")
            try:
                if backend is not None:
                    fallback_cap = cv2.VideoCapture(cam_idx, backend)
                else:
                    fallback_cap = cv2.VideoCapture(cam_idx)
                if fallback_cap.isOpened():
                    fallback_cap.set(cv2.CAP_PROP_FRAME_WIDTH, old_w)
                    fallback_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, old_h)
                    if self.hy500b_mode:
                        for key, cfg in HY500B_CONFIG["params"].items():
                            fallback_cap.set(cfg["prop"], self.param_vars[key].get())
                    with self._capture_lock:
                        self.cap = fallback_cap
                else:
                    with self._capture_lock:
                        self.cap = None
            except Exception:
                with self._capture_lock:
                    self.cap = None

            # 通知主线程切换失败
            self.root.after(0, lambda: self._on_resolution_change_done(
                success=False, actual_w=old_w, actual_h=old_h,
                actual_fps=25.0, label=label, error=str(e)))

    def _on_resolution_change_done(self, success: bool, actual_w: int, actual_h: int,
                                     actual_fps: float, label: str, error: str = "") -> None:
        """主线程回调：分辨率切换完成，恢复控件，更新UI，重启采集和预览"""
        self._resolution_changing = False
        self.resolution_combo.config(state="readonly")
        self.cam_combo.config(state="readonly")
        self.record_btn.config(state="normal")

        if success:
            if actual_w != self.current_resolution[0] or actual_h != self.current_resolution[1]:
                self.param_status_var.set(f"摄像头不支持 {label}，实际输出 {actual_w}x{actual_h}")
                self.root.after(3000, lambda: self.param_status_var.set(""))
            else:
                self.param_status_var.set(f"已切换到 {label} @ {actual_fps:.0f}fps")
                self.root.after(2000, lambda: self.param_status_var.set(""))

            mode_tag = "[HY-500B适配] " if self.hy500b_mode else ""
            self.cam_info_var.set(f"{mode_tag}已连接: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

            # 高分辨率下提示预览可能降频
            if actual_w * actual_h > 1920 * 1080:
                self.status_var.set(f"高分辨率 {actual_w}x{actual_h}，预览已自动降频以保证流畅")

            # 重启采集线程和预览
            self._start_capture_thread()
            self._start_preview()

            # 分辨率切换完成后，重新应用切换前待设置的参数（防止切换期间参数被丢弃）
            if self._pending_params:
                self.root.after(200, self._apply_pending_params)
        else:
            # 切换失败，回退下拉选择到上一个分辨率
            self.param_status_var.set(f"切换到 {label} 失败（{error[:30]}），已回退")
            self.root.after(4000, lambda: self.param_status_var.set(""))
            # 恢复下拉显示为实际分辨率
            self._param_sync_lock = True
            for i, r in enumerate(HY500B_CONFIG["resolutions"]):
                if r[0] == actual_w and r[1] == actual_h:
                    self.resolution_combo.current(i)
                    self.current_resolution = (actual_w, actual_h)
                    break
            self._param_sync_lock = False

            if self.cap is not None:
                mode_tag = "[HY-500B适配] " if self.hy500b_mode else ""
                self.cam_info_var.set(f"{mode_tag}已连接: {actual_w}x{actual_h}")
                self._start_capture_thread()
                self._start_preview()
            else:
                self.cam_info_var.set("")
                self.status_var.set("摄像头连接异常，请重新检测")

    def _reset_params(self) -> None:
        """重置参数为默认值（实时直接设置，不持锁）"""
        self._param_sync_lock = True
        failed = []
        for key, cfg in HY500B_CONFIG["params"].items():
            self.param_vars[key].set(cfg["default"])
            self.param_labels[key].config(text=f"{cfg['label']}: {cfg['default']}")
            if self.cap is not None:
                ok = self.cap.set(cfg["prop"], cfg["default"])
                if not ok:
                    failed.append(cfg["label"])
        self._param_sync_lock = False
        if failed:
            self.param_status_var.set(f"重置完成，以下参数驱动不支持: {', '.join(failed)}")
            self.root.after(4000, lambda: self.param_status_var.set(""))
        else:
            self.param_status_var.set("已重置为默认值")
            self.root.after(2000, lambda: self.param_status_var.set(""))

    def _read_params_from_camera(self) -> None:
        if self.cap is None:
            messagebox.showinfo(APP_TITLE, "请先开启摄像头预览")
            return
        self._param_sync_lock = True
        for key, cfg in HY500B_CONFIG["params"].items():
            with self._capture_lock:
                val = int(self.cap.get(cfg["prop"]))
            self.param_vars[key].set(val)
            self.param_labels[key].config(text=f"{cfg['label']}: {val}")
        self._param_sync_lock = False
        self.param_status_var.set("已读取摄像头当前参数")
        self.root.after(2000, lambda: self.param_status_var.set(""))

    def _apply_params_to_camera(self) -> None:
        if self.cap is None:
            return
        for key, cfg in HY500B_CONFIG["params"].items():
            val = self.param_vars[key].get()
            with self._capture_lock:
                self.cap.set(cfg["prop"], val)

    # ── 摄像头检测 ──
    def _on_detect(self) -> None:
        self.status_var.set("正在检测摄像头...")
        self.root.update()

        backend = self._get_backend()
        backend_name = "MSMF" if self.hy500b_mode else "默认"

        available = []
        for i in range(10):
            if backend is not None:
                cap = cv2.VideoCapture(i, backend)
            else:
                cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    available.append((i, w, h, fps))
                cap.release()

        self.available_cameras = available
        if not available:
            self.cam_combo["values"] = []
            self.cam_var.set("未检测到")
            self.cam_info_var.set("")
            self.status_var.set(f"未检测到可用摄像头（{backend_name}后端），请检查设备连接")
            messagebox.showwarning(APP_TITLE, f"未检测到可用摄像头（{backend_name}后端）")
            return

        values = [f"摄像头{idx} ({w}x{h}, {fps:.0f}fps)" for idx, w, h, fps in available]
        self.cam_combo["values"] = values
        self.cam_combo.current(0)
        self._on_camera_select(None)
        self.status_var.set(f"检测到 {len(available)} 个可用摄像头（{backend_name}后端），已自动选择第一个")

    def _on_camera_select(self, _event) -> None:
        if not self.available_cameras:
            return
        sel = self.cam_combo.current()
        if sel < 0:
            return
        idx, w, h, fps = self.available_cameras[sel]

        if self.recording:
            messagebox.showinfo(APP_TITLE, "录制中无法切换摄像头，请先停止录制")
            return

        # 停止预览和采集线程，释放旧摄像头
        self._stop_preview()
        self._stop_capture_thread()
        if self.cap is not None:
            with self._capture_lock:
                self.cap.release()
            self.cap = None

        backend = self._get_backend()
        if backend is not None:
            cap = cv2.VideoCapture(idx, backend)
        else:
            cap = cv2.VideoCapture(idx)

        if not cap.isOpened():
            messagebox.showerror(APP_TITLE, f"无法打开摄像头{idx}")
            return

        if self.hy500b_mode:
            res_w, res_h = self.current_resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            self._apply_params_to_camera()

        self.cap = cap
        self.camera_index = idx
        self._last_resolution = (w, h)  # 记录当前分辨率，用于切换失败时回退
        mode_tag = "[HY-500B适配] " if self.hy500b_mode else ""
        self.cam_info_var.set(f"{mode_tag}已连接: {w}x{h} @ {fps:.0f}fps")
        self.status_var.set(f"摄像头{idx}已开启，预览中")

        # 启动采集线程和预览
        self._start_capture_thread()
        self._start_preview()

    # ── 预览（消费者：从队列取最新帧） ──
    def _start_preview(self) -> None:
        self.preview_running = True
        self._preview_loop()

    def _stop_preview(self) -> None:
        self.preview_running = False

    def _preview_loop(self) -> None:
        if not self.preview_running:
            return
        # 从队列取最新帧（非阻塞，无帧则跳过本次刷新）
        frame = self._get_latest_frame()
        if frame is not None:
            self._show_frame(frame)

        # 动态调整预览帧率：录制时降频，高分辨率时也降频，避免画面缩放耗时导致卡顿
        if self.recording:
            interval = PREVIEW_INTERVAL_RECORDING  # ~15fps
        elif frame is not None:
            h, w = frame.shape[:2]
            pixels = w * h
            if pixels > 2048 * 1536:       # >3MP，超高分辨率
                interval = 100               # ~10fps
            elif pixels > 1920 * 1080:      # >1080p，高分辨率
                interval = 66                # ~15fps
            else:
                interval = PREVIEW_INTERVAL_NORMAL  # ~30fps
        else:
            interval = PREVIEW_INTERVAL_NORMAL
        self.root.after(interval, self._preview_loop)

    # ── 预览缩放控制 ──
    def _on_zoom_fit_toggle(self) -> None:
        """适应窗口复选框切换"""
        if self.zoom_fit_var.get():
            self.zoom_scale.config(state="disabled")
        else:
            self.zoom_scale.config(state="normal")

    def _on_zoom_change(self, _val: str) -> None:
        """缩放滑块变化，更新标签显示"""
        self.zoom_label.config(text=f"{self.zoom_var.get()}%")

    def _zoom_step(self, delta: int) -> None:
        """步进缩放（+/-按钮）"""
        if self.zoom_fit_var.get():
            # 适应窗口模式下点击+/-自动切换到自定义模式
            self.zoom_fit_var.set(False)
            self.zoom_scale.config(state="normal")
        new_val = max(10, min(300, self.zoom_var.get() + delta))
        self.zoom_var.set(new_val)
        self.zoom_label.config(text=f"{new_val}%")

    def _zoom_actual(self) -> None:
        """1:1 实际像素显示"""
        self.zoom_fit_var.set(False)
        self.zoom_scale.config(state="normal")
        self.zoom_var.set(100)
        self.zoom_label.config(text="100%")

    def _show_frame(self, frame) -> None:
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        h, w = frame.shape[:2]

        # 计算缩放比例：适应窗口模式自动计算，自定义模式用滑块值
        if self.zoom_fit_var.get():
            scale = min(cw / w, ch / h, 1.0)
        else:
            scale = self.zoom_var.get() / 100.0

        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        small = cv2.resize(frame, (nw, nh),
                           interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        # 居中显示，超出canvas部分自动被裁剪
        ix = (cw - nw) // 2
        iy = (ch - nh) // 2
        self.canvas.create_image(ix, iy, anchor="nw", image=self.photo)

    # ── 保存目录 ──
    def _on_browse(self) -> None:
        d = filedialog.askdirectory(title="选择视频保存目录")
        if d:
            self.save_dir_var.set(d)

    # ── 录制控制 ──
    def _on_toggle_record(self) -> None:
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        if self.cap is None:
            messagebox.showwarning(APP_TITLE, "请先检测并选择摄像头")
            return
        duration = self.duration_var.get()
        groups = self.groups_var.get()
        save_dir = self.save_dir_var.get().strip()
        if duration < 1:
            messagebox.showwarning(APP_TITLE, "录制时长至少 1 秒")
            return
        if groups < 1:
            messagebox.showwarning(APP_TITLE, "组数至少 1")
            return
        if not save_dir:
            messagebox.showwarning(APP_TITLE, "请选择保存目录")
            return
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        self.recording = True
        self.stop_flag = False
        self.record_btn.config(text="■ 停止录制")
        mode_tag = "[HY-500B适配] " if self.hy500b_mode else ""
        self.status_var.set(f"{mode_tag}录制中: 共{groups}组，每组{duration}秒")

        self.record_thread = threading.Thread(
            target=self._record_worker, args=(duration, groups, save_dir), daemon=True)
        self.record_thread.start()

    def _stop_recording(self) -> None:
        self.stop_flag = True
        self.recording = False
        self.record_btn.config(text="● 开始录制")
        self.status_var.set("已停止录制")
        self.progress_var.set("")

    def _record_worker(self, duration: int, groups: int, save_dir: str) -> None:
        """后台录制线程：从帧队列取帧写入视频文件，不再直接调用 cap.read()"""
        with self._capture_lock:
            if self.cap is not None:
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                fps, w, h = 25.0, 640, 480
        if fps <= 0 or fps != fps:
            fps = 25.0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_count = 0
        last_ui_update = 0.0  # UI更新限流时间戳

        for g in range(1, groups + 1):
            if self.stop_flag:
                break

            filename = f"cam{self.camera_index}_{timestamp}_g{g:02d}.mp4"
            filepath = Path(save_dir) / filename

            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(str(filepath), fourcc, fps, (w, h))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(filepath), fourcc, fps, (w, h))

            if not writer.isOpened():
                self.root.after(0, lambda: messagebox.showerror(
                    APP_TITLE, f"无法创建视频文件:\n{filepath}"))
                break

            start_time = time.time()
            frame_count = 0

            while time.time() - start_time < duration:
                if self.stop_flag:
                    break
                # 从帧队列取帧（带短超时，避免忙等占满CPU）
                try:
                    frame = self.frame_queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if frame is None:
                    continue
                writer.write(frame)
                frame_count += 1

                # UI进度更新限流：每0.2秒刷新一次，避免每帧调度导致卡顿
                now = time.time()
                if now - last_ui_update >= 0.2:
                    last_ui_update = now
                    elapsed = now - start_time
                    remain = max(0, duration - elapsed)
                    self.root.after(0, lambda g=g, groups=groups, e=elapsed, r=remain:
                        self.progress_var.set(f"第{g}/{groups}组  已录{e:.1f}s  剩余{r:.1f}s"))

            writer.release()
            saved_count += 1

            if not self.stop_flag:
                self.root.after(0, lambda fn=filename, g=g: self.status_var.set(
                    f"第{g}组已保存: {fn}"))

            # 组间短暂间隔
            time.sleep(0.3)

        self.recording = False
        self.stop_flag = False
        self.root.after(0, lambda: self._on_record_finish(saved_count, groups))

    def _on_record_finish(self, saved: int, groups: int) -> None:
        self.record_btn.config(text="● 开始录制")
        self.progress_var.set("")
        if saved == groups:
            self.status_var.set(f"录制完成！共 {groups} 组视频")
            messagebox.showinfo(APP_TITLE,
                                f"全部录制完成！\n共 {groups} 组视频\n保存目录: {self.save_dir_var.get()}")
        else:
            self.status_var.set(f"录制已停止，已保存 {saved}/{groups} 组")

    # ── 关闭清理 ──
    def _on_close(self) -> None:
        self.stop_flag = True
        self.preview_running = False
        # 先停止录制线程
        if self.record_thread is not None and self.record_thread.is_alive():
            self.record_thread.join(timeout=2)
        # 再停止采集线程
        self._stop_capture_thread()
        # 最后释放摄像头
        if self.cap is not None:
            with self._capture_lock:
                self.cap.release()
            self.cap = None
        self.root.destroy()


# ── 入口 ──
def main() -> int:
    root = tk.Tk()
    CameraRecorderApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
