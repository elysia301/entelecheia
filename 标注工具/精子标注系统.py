# -*- coding: utf-8 -*-
"""
精子标注系统（集成版）
四大模块：摄像头录制 → 视频抽帧 → 自动标注 → 人工核查
单文件，多线程，统一UI
依赖：ultralytics, opencv-python, pillow, numpy
"""
from __future__ import annotations

import re
import sys
import time
import shutil
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageTk
import cv2
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

APP_TITLE = "精子标注系统"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

# ── 模型配置 ──
MODEL_PATH = str(Path(__file__).parent / "weights" / "best.pt")
IMGSZ = 1280
DEFAULT_CONF = 0.25
IOU_THRESH = 0.4          # 黄框-红框去重IoU
REDUP_IOU_THRESH = 0.5    # 红框-红框去重IoU
MAX_AUTO_NEW_BOXES = 40   # 自动批量安全阈值
DEFAULT_AUTO_DELAY = 200  # 自动批量翻页延时(ms)

# ── 核查工具配置 ──
FIXED_BOX_W = 0.015625
FIXED_BOX_H = 0.015625
HIT_RADIUS_PX = 10


# ══════════════════════════════════════════════════════════════════════════
#  共享工具函数
# ══════════════════════════════════════════════════════════════════════════
def natural_sort_key(path_str: str) -> list:
    name = Path(path_str).name.lower()
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", name)]


def read_image_bgr(path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return img


def compute_iou(b1, b2) -> float:
    """b=(xc,yc,w,h) 归一化坐标"""
    x1_1, y1_1 = b1[0]-b1[2]/2, b1[1]-b1[3]/2
    x2_1, y2_1 = b1[0]+b1[2]/2, b1[1]+b1[3]/2
    x1_2, y1_2 = b2[0]-b2[2]/2, b2[1]-b2[3]/2
    x2_2, y2_2 = b2[0]+b2[2]/2, b2[1]+b2[3]/2
    ix1, iy1 = max(x1_1, x1_2), max(y1_1, y1_2)
    ix2, iy2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    a1 = (x2_1-x1_1)*(y2_1-y1_1)
    a2 = (x2_2-x1_2)*(y2_2-y1_2)
    union = a1+a2-inter
    return inter/union if union > 0 else 0.0


def cv2_to_photo(frame, max_w, max_h):
    h, w = frame.shape[:2]
    scale = min(max_w/w, max_h/h, 1.0)
    nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
    small = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb)), scale, nw, nh


# ══════════════════════════════════════════════════════════════════════════
#  模型推理器
# ══════════════════════════════════════════════════════════════════════════
class ModelInferencer:
    def __init__(self, model_path: str, imgsz: int = 1280):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.imgsz = imgsz

    def predict(self, image_bgr: np.ndarray, conf: float = 0.25):
        results = self.model.predict(image_bgr, imgsz=self.imgsz, conf=conf, verbose=False)
        boxes = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                h, w = image_bgr.shape[:2]
                for det in r.boxes:
                    x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
                    cx = ((x1+x2)/2) / w
                    cy = ((y1+y2)/2) / h
                    bw = (x2-x1) / w
                    bh = (y2-y1) / h
                    conf_val = float(det.conf[0].cpu().numpy())
                    cls_id = int(det.cls[0].cpu().numpy())
                    boxes.append((cls_id, cx, cy, bw, bh, conf_val))
        return boxes


# ══════════════════════════════════════════════════════════════════════════
#  模块1：摄像头录制面板
# ══════════════════════════════════════════════════════════════════════════
class RecordPanel(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.cap = None
        self.camera_index = None
        self.available_cameras = []
        self.preview_running = False
        self.photo = None
        self.recording = False
        self.record_thread = None
        self.stop_flag = False
        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Button(top, text="检测摄像头", command=self._on_detect).pack(side="left", padx=2)
        ttk.Label(top, text="摄像头:").pack(side="left", padx=(8, 2))
        self.cam_var = tk.StringVar(value="未检测")
        self.cam_combo = ttk.Combobox(top, textvariable=self.cam_var, width=28, state="readonly")
        self.cam_combo.pack(side="left", padx=2)
        self.cam_combo.bind("<<ComboboxSelected>>", self._on_select)
        self.cam_info = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.cam_info, foreground="#0066cc").pack(side="left", padx=12)

        self.canvas = tk.Canvas(self, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=4)

        bot = ttk.Frame(self)
        bot.pack(fill="x", padx=8, pady=6)
        ttk.Label(bot, text="时长(秒):").pack(side="left", padx=2)
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
        self.record_btn = ttk.Button(bot, text="● 开始录制", command=self._toggle_record)
        self.record_btn.pack(side="right", padx=2)

        status = ttk.Frame(self)
        status.pack(fill="x", padx=8, pady=(0, 4))
        self.status_var = tk.StringVar(value="请先检测摄像头")
        ttk.Label(status, textvariable=self.status_var).pack(side="left")
        self.progress_var = tk.StringVar(value="")
        ttk.Label(status, textvariable=self.progress_var, foreground="#cc3333",
                  font=("Microsoft YaHei", 10, "bold")).pack(side="right")

    def _on_detect(self):
        self.status_var.set("正在检测...")
        self.update()
        available = []
        for i in range(10):
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
            self.status_var.set("未检测到可用摄像头")
            messagebox.showwarning(APP_TITLE, "未检测到可用摄像头")
            return
        values = [f"摄像头{idx} ({w}x{h}, {fps:.0f}fps)" for idx, w, h, fps in available]
        self.cam_combo["values"] = values
        self.cam_combo.current(0)
        self._on_select(None)
        self.status_var.set(f"检测到 {len(available)} 个摄像头")

    def _on_select(self, _e):
        if not self.available_cameras or self.recording:
            return
        sel = self.cam_combo.current()
        if sel < 0:
            return
        idx, w, h, fps = self.available_cameras[sel]
        self._stop_preview()
        if self.cap:
            self.cap.release()
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            messagebox.showerror(APP_TITLE, f"无法打开摄像头{idx}")
            return
        self.cap = cap
        self.camera_index = idx
        self.cam_info.set(f"已连接: {w}x{h} @ {fps:.0f}fps")
        self.status_var.set(f"摄像头{idx}预览中")
        self._start_preview()

    def _start_preview(self):
        self.preview_running = True
        self._preview_loop()

    def _stop_preview(self):
        self.preview_running = False

    def _preview_loop(self):
        if not self.preview_running or self.cap is None:
            return
        ret, frame = self.cap.read()
        if ret and frame is not None:
            cw = max(self.canvas.winfo_width(), 100)
            ch = max(self.canvas.winfo_height(), 100)
            self.photo, _, nw, nh = cv2_to_photo(frame, cw, ch)
            self.canvas.delete("all")
            self.canvas.create_image((cw-nw)//2, (ch-nh)//2, anchor="nw", image=self.photo)
        self.after(30, self._preview_loop)

    def _on_browse(self):
        d = filedialog.askdirectory(title="选择保存目录")
        if d:
            self.save_dir_var.set(d)

    def _toggle_record(self):
        if self.recording:
            self._stop_record()
        else:
            self._start_record()

    def _start_record(self):
        if self.cap is None:
            messagebox.showwarning(APP_TITLE, "请先选择摄像头")
            return
        duration = self.duration_var.get()
        groups = self.groups_var.get()
        save_dir = self.save_dir_var.get().strip()
        if not save_dir:
            messagebox.showwarning(APP_TITLE, "请选择保存目录")
            return
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.recording = True
        self.stop_flag = False
        self.record_btn.config(text="■ 停止录制")
        self.record_thread = threading.Thread(
            target=self._record_worker, args=(duration, groups, save_dir), daemon=True)
        self.record_thread.start()

    def _stop_record(self):
        self.stop_flag = True
        self.recording = False
        self.record_btn.config(text="● 开始录制")
        self.status_var.set("已停止录制")
        self.progress_var.set("")

    def _record_worker(self, duration, groups, save_dir):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps != fps:
            fps = 25.0
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved = 0
        for g in range(1, groups+1):
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
                break
            start = time.time()
            while time.time() - start < duration:
                if self.stop_flag:
                    break
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    continue
                writer.write(frame)
                elapsed = time.time() - start
                remain = max(0, duration - elapsed)
                self.after(0, lambda g=g, groups=groups, e=elapsed, r=remain:
                    self.progress_var.set(f"第{g}/{groups}组  已录{e:.1f}s  剩余{r:.1f}s"))
            writer.release()
            saved += 1
            if not self.stop_flag:
                self.after(0, lambda fn=filename: self.status_var.set(f"已保存: {fn}"))
            time.sleep(0.3)
        self.recording = False
        self.stop_flag = False
        self.after(0, lambda: self._on_finish(saved, groups))

    def _on_finish(self, saved, groups):
        self.record_btn.config(text="● 开始录制")
        self.progress_var.set("")
        if saved == groups:
            self.status_var.set(f"录制完成！共{groups}组")
            messagebox.showinfo(APP_TITLE, f"全部录制完成！\n共{groups}组视频\n目录: {self.save_dir_var.get()}")
        else:
            self.status_var.set(f"已停止，保存{saved}/{groups}组")

    def cleanup(self):
        self.stop_flag = True
        self.preview_running = False
        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=2)
        if self.cap:
            self.cap.release()


# ══════════════════════════════════════════════════════════════════════════
#  模块2：视频抽帧面板
# ══════════════════════════════════════════════════════════════════════════
class ExtractPanel(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.video_path = None
        self.cap = None
        self._tmp_path = None
        self.total_frames = 0
        self.fps = 0.0
        self.duration = 0.0
        self.width = 0
        self.height = 0
        self.photo = None
        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Button(top, text="导入视频", command=self._on_import).pack(side="left", padx=2)
        self.info_var = tk.StringVar(value="未导入视频")
        ttk.Label(top, textvariable=self.info_var, font=("Microsoft YaHei", 10)).pack(side="left", padx=12)

        self.canvas = tk.Canvas(self, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=4)

        bot = ttk.Frame(self)
        bot.pack(fill="x", padx=8, pady=6)
        ttk.Label(bot, text="抽取数量:").pack(side="left", padx=2)
        self.count_var = tk.IntVar(value=100)
        self.count_spin = ttk.Spinbox(bot, from_=1, to=99999, width=8, textvariable=self.count_var)
        self.count_spin.pack(side="left", padx=2)
        ttk.Label(bot, text="张（均匀间隔）").pack(side="left", padx=2)
        ttk.Separator(bot, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(bot, text="预览帧:").pack(side="left", padx=2)
        self.preview_var = tk.IntVar(value=0)
        self.preview_scale = ttk.Scale(bot, from_=0, to=0, orient="horizontal", length=200,
                                         variable=self.preview_var, command=self._on_preview)
        self.preview_scale.pack(side="left", padx=2)
        self.preview_label = ttk.Label(bot, text="0 / 0")
        self.preview_label.pack(side="left", padx=4)
        ttk.Separator(bot, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(bot, text="导出图片", command=self._on_export).pack(side="right", padx=2)

        self.status_var = tk.StringVar(value="请导入视频")
        ttk.Label(self, textvariable=self.status_var, foreground="#333").pack(anchor="w", padx=8, pady=(0, 4))

    def _open_video(self, path):
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            return cap
        suffix = Path(path).suffix
        tmp = Path(tempfile.gettempdir()) / f"_extract_tmp{suffix}"
        try:
            shutil.copy2(path, tmp)
        except Exception:
            return None
        cap = cv2.VideoCapture(str(tmp))
        if cap.isOpened():
            self._tmp_path = tmp
            return cap
        return None

    def _release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        if self._tmp_path and self._tmp_path.exists():
            try:
                self._tmp_path.unlink()
            except Exception:
                pass
            self._tmp_path = None

    def _on_import(self):
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", " ".join(f"*{e}" for e in sorted(VIDEO_EXTS))), ("所有文件", "*.*")])
        if not path:
            return
        self._release()
        cap = self._open_video(path)
        if cap is None or not cap.isOpened():
            messagebox.showerror(APP_TITLE, f"无法打开视频:\n{path}")
            return
        self.video_path = path
        self.cap = cap
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0 or not np.isfinite(self.fps):
            self.fps = 30.0
        self.duration = self.total_frames / self.fps
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_sec = int(self.duration)
        h, rem = divmod(total_sec, 3600)
        m, s = divmod(rem, 60)
        dur_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
        self.info_var.set(f"{Path(path).name}  |  {self.width}×{self.height}  |  "
                          f"{self.total_frames}帧  |  {self.fps:.1f}fps  |  时长 {dur_str}")
        max_f = max(0, self.total_frames - 1)
        self.preview_scale.config(to=max_f)
        self.preview_var.set(0)
        self.count_spin.config(to=max(1, self.total_frames))
        if self.count_var.get() > self.total_frames:
            self.count_var.set(self.total_frames)
        self.preview_label.config(text=f"0 / {max_f}")
        self.status_var.set("视频已加载，可调节抽取数量后导出")
        self._show_preview()

    def _on_preview(self, _v):
        if self.total_frames > 0:
            self.preview_label.config(text=f"{self.preview_var.get()} / {self.total_frames - 1}")
        self._show_preview()

    def _show_preview(self):
        if self.cap is None or self.total_frames == 0:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.preview_var.get())
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        self.photo, _, nw, nh = cv2_to_photo(frame, cw, ch)
        self.canvas.delete("all")
        self.canvas.create_image((cw-nw)//2, (ch-nh)//2, anchor="nw", image=self.photo)

    def _on_export(self):
        if self.cap is None:
            messagebox.showwarning(APP_TITLE, "请先导入视频")
            return
        count = self.count_var.get()
        if count < 1 or count > self.total_frames:
            messagebox.showwarning(APP_TITLE, f"抽取数量范围 1~{self.total_frames}")
            return
        out_dir = filedialog.askdirectory(title="选择导出目录")
        if not out_dir:
            return
        indices = np.linspace(0, self.total_frames - 1, count, dtype=int)
        video_stem = Path(self.video_path).stem
        out_path = Path(out_dir)

        prog = tk.Toplevel(self)
        prog.title("导出中...")
        prog.geometry("360x110")
        prog.transient(self)
        prog.grab_set()
        ttk.Label(prog, text=f"正在导出 {count} 张图片...").pack(pady=8)
        bar = ttk.Progressbar(prog, length=320, mode="determinate", maximum=count)
        bar.pack(pady=4)
        lbl = ttk.Label(prog, text=f"0 / {count}")
        lbl.pack()

        def worker():
            exported, failed = 0, 0
            for i, idx in enumerate(indices):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    filename = f"{video_stem}_f{int(idx):04d}.png"
                    ok, buf = cv2.imencode(".png", frame)
                    if ok:
                        buf.tofile(str(out_path / filename))
                        exported += 1
                    else:
                        failed += 1
                else:
                    failed += 1
                self.after(0, lambda i=i+1: (bar.config(value=i+1), lbl.config(text=f"{i+1} / {count}")))
            self.after(0, lambda: (prog.destroy(), self._on_export_done(exported, failed, out_dir)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_export_done(self, exported, failed, out_dir):
        self.status_var.set(f"导出完成: {exported}张成功, {failed}张失败")
        msg = f"导出完成！\n成功: {exported} 张\n失败: {failed} 张\n目录: {out_dir}"
        if failed > 0:
            msg += "\n\n（失败通常是视频末尾损坏帧）"
        messagebox.showinfo(APP_TITLE, msg)

    def cleanup(self):
        self._release()


# ══════════════════════════════════════════════════════════════════════════
#  模块3：自动标注面板
# ══════════════════════════════════════════════════════════════════════════
class AnnotatePanel(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.inferencer = None
        self.images = []
        self.img_dir = None
        self.lbl_dir = None
        self.idx = 0
        self.img = None
        self.gt_boxes = []
        self.pred_boxes = []
        self.deleted_pred = set()
        self.dirty = False
        self.auto_mode = False
        self.auto_log = []
        self.skipped_images = []
        self.auto_removed_count = 0
        self.gg_removed_count = 0
        self.conf_threshold = DEFAULT_CONF
        self.iou_thresh_gp = IOU_THRESH
        self.iou_thresh_gg = REDUP_IOU_THRESH
        self.auto_delay = DEFAULT_AUTO_DELAY
        self.photo = None
        self._scale = 1.0
        self._pad_x = 0
        self._pad_y = 0
        self._disp_w = 640
        self._disp_h = 640
        self._build_ui()

    def _build_ui(self):
        # 顶部参数栏
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=6, pady=4)
        ttk.Button(bar, text="导入数据", command=self._on_import).pack(side="left", padx=2)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Label(bar, text="置信度:").pack(side="left", padx=2)
        self.conf_var = tk.StringVar(value=f"{DEFAULT_CONF:.2f}")
        ttk.Entry(bar, textvariable=self.conf_var, width=5).pack(side="left", padx=2)
        ttk.Label(bar, text="黄红IoU:", foreground="#CC9900").pack(side="left", padx=(8, 2))
        self.iou_gp_var = tk.StringVar(value=f"{IOU_THRESH:.2f}")
        ttk.Entry(bar, textvariable=self.iou_gp_var, width=5).pack(side="left", padx=2)
        ttk.Label(bar, text="红红IoU:", foreground="#CC3333").pack(side="left", padx=(8, 2))
        self.iou_gg_var = tk.StringVar(value=f"{REDUP_IOU_THRESH:.2f}")
        ttk.Entry(bar, textvariable=self.iou_gg_var, width=5).pack(side="left", padx=2)
        ttk.Label(bar, text="延时ms:", foreground="#3366CC").pack(side="left", padx=(8, 2))
        self.auto_delay_var = tk.StringVar(value=str(DEFAULT_AUTO_DELAY))
        ttk.Entry(bar, textvariable=self.auto_delay_var, width=5).pack(side="left", padx=2)
        ttk.Button(bar, text="应用", width=6, command=self._apply_thresholds).pack(side="left", padx=(8, 2))
        ttk.Button(bar, text="重新推理", width=8, command=self._re_infer).pack(side="left", padx=4)
        ttk.Button(bar, text="💾 保存(W)", width=12, command=self._save).pack(side="right", padx=6)
        self.auto_btn = ttk.Button(bar, text="▶ 自动批量", width=12, command=self._toggle_auto)
        self.auto_btn.pack(side="right", padx=6)
        self.count_label = ttk.Label(bar, text="")
        self.count_label.pack(side="right", padx=10)

        # 画布
        self.canvas = tk.Canvas(self, bg="#202020", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=6, pady=4)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Configure>", lambda e: self._on_resize())

        # 底部导航
        bot = ttk.Frame(self)
        bot.pack(fill="x", padx=6, pady=4)
        ttk.Button(bot, text="◀ 上一张(A)", command=self._prev).pack(side="left", padx=2)
        ttk.Button(bot, text="下一张 ▶(D)", command=self._next).pack(side="left", padx=2)
        ttk.Button(bot, text="取消(S)", command=self._cancel).pack(side="left", padx=2)
        ttk.Separator(bot, orient="vertical").pack(side="left", fill="y", padx=6)
        self.nav_label = ttk.Label(bot, text="未导入数据", font=("Microsoft YaHei", 10))
        self.nav_label.pack(side="left", padx=6)
        hint = ("左键点框=删除 ｜ 左键点空白=新增红框 ｜ A/D=翻页 ｜ S=取消 ｜ W=保存 ｜ "
                "自动批量逐张保存")
        ttk.Label(bot, text=hint, foreground="#666", font=("Microsoft YaHei", 9)).pack(side="right", padx=6)

        # 键盘
        self.bind_all("<a>", lambda e: self._prev())
        self.bind_all("<d>", lambda e: self._next())
        self.bind_all("<s>", lambda e: self._cancel())
        self.bind_all("<w>", lambda e: self._save())
        self.bind_all("<A>", lambda e: self._prev())
        self.bind_all("<D>", lambda e: self._next())
        self.bind_all("<S>", lambda e: self._cancel())
        self.bind_all("<W>", lambda e: self._save())

    # ── 导入 ──
    def _on_import(self):
        if self.auto_mode:
            messagebox.showinfo(APP_TITLE, "自动模式运行中，请先停止")
            return
        img_dir = filedialog.askdirectory(title="选择图片目录")
        if not img_dir:
            return
        lbl_dir = filedialog.askdirectory(title="选择YOLO标签目录")
        if not lbl_dir:
            return
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.images = [f for f in self.img_dir.iterdir()
                       if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        self.images.sort(key=lambda p: natural_sort_key(str(p)))
        if not self.images:
            messagebox.showwarning(APP_TITLE, "图片目录中没有图片")
            return
        # 加载模型
        if self.inferencer is None:
            if not Path(MODEL_PATH).exists():
                mp = filedialog.askopenfilename(title="选择模型文件(best.pt)",
                                                 filetypes=[("PyTorch模型", "*.pt"), ("所有文件", "*.*")])
                if not mp:
                    return
                self._model_path = mp
            else:
                self._model_path = MODEL_PATH
            try:
                self.inferencer = ModelInferencer(self._model_path, IMGSZ)
            except Exception as e:
                messagebox.showerror(APP_TITLE, f"模型加载失败:\n{e}")
                return
        self.idx = 0
        self.auto_log = []
        self.skipped_images = []
        self._load_index(0)
        self.app.set_status(f"已加载 {len(self.images)} 张图片，模型就绪")

    def _label_path(self, img_path):
        return self.lbl_dir / f"{img_path.stem}.txt"

    def _parse_labels(self, label_path):
        boxes = []
        if not label_path.exists():
            return boxes
        try:
            content = label_path.read_text(encoding="utf-8").strip()
        except Exception:
            return boxes
        for line in content.splitlines():
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except (ValueError, IndexError):
                continue
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                continue
            boxes.append((cls, cx, cy, w, h))
        return boxes

    def _dedupe_red(self, boxes):
        """红红去重：IoU超过阈值的重复红框，保留面积较大的"""
        if len(boxes) < 2:
            return boxes, 0
        removed = 0
        keep = list(boxes)
        changed = True
        while changed:
            changed = False
            for i in range(len(keep)):
                for j in range(i+1, len(keep)):
                    if compute_iou(keep[i][1:], keep[j][1:]) >= self.iou_thresh_gg:
                        ai = keep[i][3] * keep[i][4]
                        aj = keep[j][3] * keep[j][4]
                        remove_idx = j if ai >= aj else i
                        keep.pop(remove_idx)
                        removed += 1
                        changed = True
                        break
                if changed:
                    break
        return keep, removed

    def _dedupe_yellow(self, pred_boxes, gt_boxes):
        """黄红去重：与红框IoU超过阈值的黄框删除"""
        keep = []
        removed = 0
        for pb in pred_boxes:
            overlap = False
            for gb in gt_boxes:
                if compute_iou(pb[1:5], gb[1:5]) >= self.iou_thresh_gp:
                    overlap = True
                    break
            if overlap:
                removed += 1
            else:
                keep.append(pb)
        return keep, removed

    def _load_index(self, i):
        if not self.images:
            return
        self.idx = max(0, min(i, len(self.images) - 1))
        img_path = self.images[self.idx]
        try:
            self.img = read_image_bgr(img_path)
        except Exception as e:
            messagebox.showerror("读取失败", str(e))
            return
        # 加载标签 + 红红去重
        raw_gt = self._parse_labels(self._label_path(img_path))
        self.gt_boxes, self.gg_removed_count = self._dedupe_red(raw_gt)
        # 模型推理 + 黄红去重
        try:
            raw_pred = self.inferencer.predict(self.img, self.conf_threshold)
        except Exception as e:
            messagebox.showerror("推理失败", str(e))
            raw_pred = []
        self.pred_boxes, self.auto_removed_count = self._dedupe_yellow(raw_pred, self.gt_boxes)
        self.deleted_pred = set()
        self.dirty = False
        self._update_status()
        self._redraw()

    def _update_status(self):
        if not self.images:
            return
        img_path = self.images[self.idx]
        exists = "✅" if self._label_path(img_path).exists() else "（无标签，保存将新建）"
        self.nav_label.config(text=f"[{self.idx+1}/{len(self.images)}] {img_path.name}  {exists}")
        active_pred = len(self.pred_boxes) - len(self.deleted_pred)
        mark = " ●未保存" if self.dirty else ""
        auto_info = f" 黄红去重:{self.auto_removed_count}" if self.auto_removed_count > 0 else ""
        gg_info = f" 红红去重:{self.gg_removed_count}" if self.gg_removed_count > 0 else ""
        mode_info = "  [自动模式]" if self.auto_mode else ""
        self.count_label.config(text=f"红框:{len(self.gt_boxes)}  黄框:{active_pred}/{len(self.pred_boxes)}"
                                      f"{gg_info}{auto_info}{mode_info}{mark}")

    def _redraw(self):
        self.canvas.delete("all")
        if self.img is None:
            return
        cw = max(self.canvas.winfo_width(), 10)
        ch = max(self.canvas.winfo_height(), 10)
        h, w = self.img.shape[:2]
        self._scale = min(cw/w, ch/h)
        dw, dh = int(w*self._scale), int(h*self._scale)
        self._pad_x = (cw - dw) // 2
        self._pad_y = (ch - dh) // 2
        self.photo, _, _, _ = cv2_to_photo(self.img, cw, ch)
        self.canvas.create_image(self._pad_x, self._pad_y, anchor="nw", image=self.photo)
        # 红框
        for cls, cx, cy, bw, bh in self.gt_boxes:
            x1 = self._pad_x + (cx - bw/2) * dw
            y1 = self._pad_y + (cy - bh/2) * dh
            x2 = self._pad_x + (cx + bw/2) * dw
            y2 = self._pad_y + (cy + bh/2) * dh
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#FF2020", width=2)
        # 黄框
        for i, (cls, cx, cy, bw, bh, conf) in enumerate(self.pred_boxes):
            if i in self.deleted_pred:
                continue
            x1 = self._pad_x + (cx - bw/2) * dw
            y1 = self._pad_y + (cy - bh/2) * dh
            x2 = self._pad_x + (cx + bw/2) * dw
            y2 = self._pad_y + (cy + bh/2) * dh
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="#FFCC00", width=2)

    def _on_resize(self):
        if self.img is not None:
            self._redraw()

    def _canvas_to_img(self, ex, ey):
        if self.img is None:
            return None
        h, w = self.img.shape[:2]
        dw, dh = int(w*self._scale), int(h*self._scale)
        mx = ex - self._pad_x
        my = ey - self._pad_y
        if not (0 <= mx < dw and 0 <= my < dh):
            return None
        return mx/dw, my/dh

    def _on_click(self, event):
        if self.img is None or self.auto_mode:
            return
        coord = self._canvas_to_img(event.x, event.y)
        if coord is None:
            return
        cx, cy = coord
        # 先检查是否点中黄框
        for i, (cls, bx, by, bw, bh, conf) in enumerate(self.pred_boxes):
            if i in self.deleted_pred:
                continue
            if abs(cx - bx) < bw/2 and abs(cy - by) < bh/2:
                self.deleted_pred.add(i)
                self.dirty = True
                self._update_status()
                self._redraw()
                return
        # 检查是否点中红框
        for i, (cls, bx, by, bw, bh) in enumerate(self.gt_boxes):
            if abs(cx - bx) < bw/2 and abs(cy - by) < bh/2:
                self.gt_boxes.pop(i)
                self.dirty = True
                self._update_status()
                self._redraw()
                return
        # 空白处新增红框（固定10x10像素）
        h, w = self.img.shape[:2]
        self.gt_boxes.append((0, cx, cy, 10/w, 10/h))
        self.dirty = True
        self._update_status()
        self._redraw()

    def _prev(self):
        if not self.images or self.auto_mode:
            return
        if self.dirty:
            ans = messagebox.askyesnocancel("未保存", "当前有未保存修改，是否保存后翻页？")
            if ans is None:
                return
            if ans:
                self._save(silent=True)
        self._load_index(self.idx - 1)

    def _next(self):
        if not self.images or self.auto_mode:
            return
        if self.dirty:
            ans = messagebox.askyesnocancel("未保存", "当前有未保存修改，是否保存后翻页？")
            if ans is None:
                return
            if ans:
                self._save(silent=True)
        self._load_index(self.idx + 1)

    def _cancel(self):
        if not self.images:
            return
        self._load_index(self.idx)
        self.app.set_status("已取消当前修改")

    def _save(self, silent=False):
        if not self.images:
            return
        img_path = self.images[self.idx]
        label_path = self._label_path(img_path)
        # 合并：红框 + 未删除的黄框
        all_boxes = list(self.gt_boxes)
        for i, pb in enumerate(self.pred_boxes):
            if i not in self.deleted_pred:
                all_boxes.append((pb[0], pb[1], pb[2], pb[3], pb[4]))
        lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in all_boxes]
        try:
            self.lbl_dir.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))
            return
        self.dirty = False
        self._update_status()
        if not silent:
            self.app.set_status(f"已保存: {label_path.name} ({len(all_boxes)}个框)")

    def _apply_thresholds(self):
        if self.auto_mode:
            messagebox.showinfo(APP_TITLE, "自动模式运行中，请先停止再调节阈值")
            return
        try:
            conf = float(self.conf_var.get())
            iou_gp = float(self.iou_gp_var.get())
            iou_gg = float(self.iou_gg_var.get())
            auto_delay = int(float(self.auto_delay_var.get()))
        except ValueError:
            messagebox.showwarning("输入错误", "参数必须是数字")
            return
        if not (0 < conf < 1):
            messagebox.showwarning("输入错误", "置信度范围 0~1")
            return
        if not (0 <= iou_gp <= 1 and 0 <= iou_gg <= 1):
            messagebox.showwarning("输入错误", "IoU范围 0~1")
            return
        if not (10 <= auto_delay <= 10000):
            messagebox.showwarning("输入错误", "延时范围 10~10000ms")
            return
        self.conf_threshold = conf
        self.iou_thresh_gp = iou_gp
        self.iou_thresh_gg = iou_gg
        self.auto_delay = auto_delay
        if self.images:
            self._load_index(self.idx)
        self.app.set_status(f"阈值已应用: conf={conf}, 黄红IoU={iou_gp}, 红红IoU={iou_gg}, 延时={auto_delay}ms")

    def _re_infer(self):
        if self.auto_mode or not self.images:
            return
        try:
            conf = float(self.conf_var.get())
            if 0 < conf < 1:
                self.conf_threshold = conf
        except ValueError:
            pass
        self._load_index(self.idx)
        self.app.set_status("已重新推理当前帧")

    # ── 自动批量 ──
    def _toggle_auto(self):
        if not self.images:
            messagebox.showwarning(APP_TITLE, "请先导入数据")
            return
        if self.auto_mode:
            self.auto_mode = False
            self.auto_btn.config(text="▶ 自动批量")
            self.app.set_status("已停止自动批量模式")
            if self.auto_log:
                self._show_auto_log()
            return
        # 启动
        self.auto_mode = True
        self.auto_log = []
        self.skipped_images = []
        self.auto_btn.config(text="⏹ 停止自动")
        self.app.set_status(f"自动批量模式启动（置信度={self.conf_threshold:.2f}，延时={self.auto_delay}ms）")
        self.after(self.auto_delay, self._auto_step)

    def _auto_step(self):
        if not self.auto_mode or not self.images:
            return
        if self.idx >= len(self.images):
            self.auto_mode = False
            self.auto_btn.config(text="▶ 自动批量")
            self.app.set_status("自动批量完成！")
            self._show_auto_log()
            return
        img_path = self.images[self.idx]
        has_label = self._label_path(img_path).exists()
        original_gt_count = len(self.gt_boxes)
        # 新增黄框数
        new_count = len(self.pred_boxes) - len(self.deleted_pred)
        # 安全检查：已有标签的图片，单次新增超过阈值则跳过
        skipped = False
        if has_label and new_count >= MAX_AUTO_NEW_BOXES:
            skipped = True
            self.skipped_images.append(img_path.name)
        else:
            # 保存
            all_boxes = list(self.gt_boxes)
            for i, pb in enumerate(self.pred_boxes):
                if i not in self.deleted_pred:
                    all_boxes.append((pb[0], pb[1], pb[2], pb[3], pb[4]))
            lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in all_boxes]
            try:
                self.lbl_dir.mkdir(parents=True, exist_ok=True)
                self._label_path(img_path).write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            except Exception as e:
                self.app.set_status(f"保存失败: {e}")
        self.auto_log.append({
            "name": img_path.name, "idx": self.idx,
            "original": original_gt_count, "new": new_count if not skipped else 0,
            "total": original_gt_count + (new_count if not skipped else 0),
            "action": "跳过" if skipped else "保存",
            "note": f"新增{new_count}≥{MAX_AUTO_NEW_BOXES}" if skipped else ""
        })
        self.app.set_status(f"自动: [{self.idx+1}/{len(self.images)}] {img_path.name} "
                            f"→ {'跳过(异常)' if skipped else '已保存'}")
        # 下一张
        self.idx += 1
        if self.idx < len(self.images):
            self._load_index(self.idx)
        self.after(self.auto_delay, self._auto_step)

    def _show_auto_log(self):
        win = tk.Toplevel(self)
        win.title("自动批量处理日志")
        win.geometry("900x500")
        saved = [l for l in self.auto_log if l["action"] == "保存"]
        skipped = [l for l in self.auto_log if l["action"] == "跳过"]
        total_new = sum(l["new"] for l in saved)
        total_boxes = sum(l["total"] for l in saved)
        summary = (f"处理总数: {len(self.auto_log)}  |  成功: {len(saved)}  |  跳过: {len(skipped)}\n"
                   f"新增坐标: {total_new}  |  保存后总坐标: {total_boxes}\n"
                   f"置信度: {self.conf_threshold:.2f}  |  黄红IoU: {self.iou_thresh_gp:.2f}  |  "
                   f"红红IoU: {self.iou_thresh_gg:.2f}  |  延时: {self.auto_delay}ms")
        ttk.Label(win, text=summary, justify="left", font=("Consolas", 10)).pack(anchor="w", padx=10, pady=6)
        txt = tk.Text(win, font=("Consolas", 9), wrap="none")
        txt.pack(fill="both", expand=True, padx=10, pady=4)
        sb = ttk.Scrollbar(win, orient="vertical", command=txt.yview)
        txt.config(yscrollcommand=sb.set)
        header = f"{'序号':<6}{'文件名':<45}{'操作':<6}{'原有':<6}{'新增':<6}{'总数':<6}{'备注'}\n"
        txt.insert("end", header)
        txt.insert("end", "-" * 90 + "\n")
        for log in self.auto_log:
            line = (f"{log['idx']+1:<6}{log['name'][:43]:<45}{log['action']:<6}"
                    f"{log['original']:<6}{log['new']:<6}{log['total']:<6}{log['note']}\n")
            txt.insert("end", line)
        txt.config(state="disabled")

    def cleanup(self):
        self.auto_mode = False


# ══════════════════════════════════════════════════════════════════════════
#  模块4：人工核查面板
# ══════════════════════════════════════════════════════════════════════════
class ReviewPanel(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.images = []
        self.img_dir = None
        self.lbl_dir = None
        self.idx = 0
        self.img = None
        self.boxes = []
        self.dirty = False
        self.photo = None
        self._scale = 1.0
        self._off_x = 0
        self._off_y = 0
        self._build_ui()

    def _build_ui(self):
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=6, pady=4)
        ttk.Button(bar, text="导入数据", command=self._on_import).pack(side="left", padx=2)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(bar, text="◀ 上一张", command=self._prev).pack(side="left", padx=2)
        ttk.Button(bar, text="下一张 ▶", command=self._next).pack(side="left", padx=2)
        self.nav_label = ttk.Label(bar, text="未导入数据", font=("Microsoft YaHei", 10))
        self.nav_label.pack(side="left", padx=14)
        ttk.Button(bar, text="💾 导出覆写", command=self._save).pack(side="right", padx=6)
        self.count_label = ttk.Label(bar, text="")
        self.count_label.pack(side="right", padx=14)

        self.canvas = tk.Canvas(self, bg="#202020", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=6, pady=4)
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Configure>", lambda e: self._redraw())

        hint = "操作：左键点空白=新增框 ｜ 左键点中已有框=删除 ｜ ←/→ 翻页 ｜ 导出覆写当前标签"
        ttk.Label(self, text=hint, foreground="#666", font=("Microsoft YaHei", 9)).pack(anchor="w", padx=8, pady=(0, 4))

        self.bind_all("<Left>", lambda e: self._prev())
        self.bind_all("<Right>", lambda e: self._next())

    def _on_import(self):
        img_dir = filedialog.askdirectory(title="选择图片目录")
        if not img_dir:
            return
        lbl_dir = filedialog.askdirectory(title="选择YOLO标签目录")
        if not lbl_dir:
            return
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.images = [f for f in self.img_dir.iterdir()
                       if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        self.images.sort(key=lambda p: natural_sort_key(str(p)))
        if not self.images:
            messagebox.showwarning(APP_TITLE, "图片目录中没有图片")
            return
        self.idx = 0
        self._load_index(0)
        self.app.set_status(f"已加载 {len(self.images)} 张图片，核查模式")

    def _label_path(self, img_path):
        return self.lbl_dir / f"{img_path.stem}.txt"

    def _parse_labels(self, label_path):
        boxes = []
        if not label_path.exists():
            return boxes
        try:
            content = label_path.read_text(encoding="utf-8").strip()
        except Exception:
            return boxes
        for line in content.splitlines():
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy = float(parts[1]), float(parts[2])
            except (ValueError, IndexError):
                continue
            if not (0 <= cx <= 1 and 0 <= cy <= 1):
                continue
            boxes.append((cls, cx, cy))
        return boxes

    def _load_index(self, i):
        if not self.images:
            return
        self.idx = max(0, min(i, len(self.images) - 1))
        img_path = self.images[self.idx]
        try:
            self.img = read_image_bgr(img_path)
        except Exception as e:
            messagebox.showerror("读取失败", str(e))
            return
        self.boxes = self._parse_labels(self._label_path(img_path))
        self.dirty = False
        self._update_status()
        self._redraw()

    def _update_status(self):
        if not self.images:
            return
        img_path = self.images[self.idx]
        exists = "✅" if self._label_path(img_path).exists() else "（无标签，将新建）"
        self.nav_label.config(text=f"[{self.idx+1}/{len(self.images)}] {img_path.name}  {exists}")
        mark = " ●未保存" if self.dirty else ""
        self.count_label.config(text=f"框数: {len(self.boxes)}{mark}")

    def _redraw(self):
        self.canvas.delete("all")
        if self.img is None:
            return
        cw = max(self.canvas.winfo_width(), 10)
        ch = max(self.canvas.winfo_height(), 10)
        h, w = self.img.shape[:2]
        self._scale = min(cw/w, ch/h)
        dw, dh = int(w*self._scale), int(h*self._scale)
        self._off_x = (cw - dw) // 2
        self._off_y = (ch - dh) // 2
        self.photo, _, _, _ = cv2_to_photo(self.img, cw, ch)
        self.canvas.create_image(self._off_x, self._off_y, anchor="nw", image=self.photo)
        box_w = FIXED_BOX_W * w * self._scale
        box_h = FIXED_BOX_H * h * self._scale
        for _, cx, cy in self.boxes:
            x = self._off_x + cx * dw
            y = self._off_y + cy * dh
            self.canvas.create_rectangle(x-box_w/2, y-box_h/2, x+box_w/2, y+box_h/2,
                                         outline="#FF2020", width=2)

    def _on_click(self, event):
        if self.img is None:
            return
        h, w = self.img.shape[:2]
        dw, dh = int(w*self._scale), int(h*self._scale)
        mx = event.x - self._off_x
        my = event.y - self._off_y
        if not (0 <= mx < dw and 0 <= my < dh):
            return
        cx, cy = mx/dw, my/dh
        hit_r = max(HIT_RADIUS_PX, FIXED_BOX_W * w * self._scale)
        best_i, best_d = -1, hit_r
        for i, (_, bx, by) in enumerate(self.boxes):
            dx = (bx - cx) * dw
            dy = (by - cy) * dh
            d = (dx*dx + dy*dy) ** 0.5
            if d < best_d:
                best_i, best_d = i, d
        if best_i >= 0:
            self.boxes.pop(best_i)
        else:
            self.boxes.append((0, cx, cy))
        self.dirty = True
        self._update_status()
        self._redraw()

    def _prev(self):
        if not self.images:
            return
        if self.dirty:
            ans = messagebox.askyesnocancel("未保存", "当前有未保存修改，是否保存后翻页？")
            if ans is None:
                return
            if ans:
                self._save(silent=True)
        self._load_index(self.idx - 1)

    def _next(self):
        if not self.images:
            return
        if self.dirty:
            ans = messagebox.askyesnocancel("未保存", "当前有未保存修改，是否保存后翻页？")
            if ans is None:
                return
            if ans:
                self._save(silent=True)
        self._load_index(self.idx + 1)

    def _save(self, silent=False):
        if not self.images:
            return
        img_path = self.images[self.idx]
        label_path = self._label_path(img_path)
        lines = [f"{c} {x:.6f} {y:.6f} {FIXED_BOX_W:.6f} {FIXED_BOX_H:.6f}"
                 for c, x, y in self.boxes]
        try:
            self.lbl_dir.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))
            return
        self.dirty = False
        self._update_status()
        if not silent:
            self.app.set_status(f"已保存: {label_path.name} ({len(self.boxes)}个框)")

    def cleanup(self):
        pass


# ══════════════════════════════════════════════════════════════════════════
#  主应用：左侧导航 + 面板切换
# ══════════════════════════════════════════════════════════════════════════
class MainApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1200x850")
        self.root.minsize(900, 650)

        # 左侧导航
        self.nav = ttk.Frame(root, width=160)
        self.nav.pack(side="left", fill="y")
        self.nav.pack_propagate(False)

        ttk.Label(self.nav, text=APP_TITLE, font=("Microsoft YaHei", 14, "bold"),
                  foreground="#0066cc").pack(pady=(16, 8))
        ttk.Separator(self.nav, orient="horizontal").pack(fill="x", padx=10, pady=4)

        self.panels = {}
        self.nav_buttons = {}
        modules = [
            ("record", "📹 摄像头录制", RecordPanel),
            ("extract", "🎬 视频抽帧", ExtractPanel),
            ("annotate", "🏷️ 自动标注", AnnotatePanel),
            ("review", "✅ 人工核查", ReviewPanel),
        ]
        for key, label, panel_cls in modules:
            btn = ttk.Button(self.nav, text=label, command=lambda k=key: self._switch(k))
            btn.pack(fill="x", padx=8, pady=3)
            self.nav_buttons[key] = btn

        ttk.Separator(self.nav, orient="horizontal").pack(fill="x", padx=10, pady=(8, 4))
        ttk.Label(self.nav, text="工作流顺序:", font=("Microsoft YaHei", 9),
                  foreground="#888").pack(anchor="w", padx=12, pady=2)
        ttk.Label(self.nav, text="1.录制 → 2.抽帧\n3.标注 → 4.核查",
                  font=("Microsoft YaHei", 9), foreground="#666", justify="left").pack(anchor="w", padx=12)

        # 右侧主内容区
        self.main = ttk.Frame(root)
        self.main.pack(side="left", fill="both", expand=True)

        # 底部状态栏
        self.status_bar = ttk.Frame(root)
        self.status_bar.pack(side="bottom", fill="x")
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(self.status_bar, textvariable=self.status_var, relief="sunken",
                  anchor="w", padding=(8, 2)).pack(fill="x")

        # 创建所有面板
        self.container = ttk.Frame(self.main)
        self.container.pack(fill="both", expand=True)
        for key, label, panel_cls in modules:
            panel = panel_cls(self.container, self)
            self.panels[key] = panel

        self.current = None
        self._switch("record")

    def _switch(self, key):
        if self.current == key:
            return
        # 隐藏当前
        if self.current and self.current in self.panels:
            self.panels[self.current].pack_forget()
        # 显示新的
        self.panels[key].pack(fill="both", expand=True)
        self.current = key
        # 更新导航按钮状态
        for k, btn in self.nav_buttons.items():
            if k == key:
                btn.state(["pressed"])
            else:
                btn.state(["!pressed"])
        self.set_status(f"当前模块: {self._module_name(key)}")

    def _module_name(self, key):
        names = {"record": "摄像头录制", "extract": "视频抽帧",
                 "annotate": "自动标注", "review": "人工核查"}
        return names.get(key, key)

    def set_status(self, msg):
        self.status_var.set(msg)

    def cleanup(self):
        for panel in self.panels.values():
            if hasattr(panel, "cleanup"):
                panel.cleanup()


# ══════════════════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════════════════
def main() -> int:
    root = tk.Tk()
    app = MainApp(root)

    def _on_close():
        app.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
