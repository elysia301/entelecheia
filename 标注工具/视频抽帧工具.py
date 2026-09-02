# -*- coding: utf-8 -*-
"""
视频抽帧工具 v3（批量版）
功能：
  - 单视频模式：导入单个视频 → 预览 → 设置参数 → 导出
  - 批量模式：多选视频文件/导入文件夹 → 视频列表 → 统一设置抽帧参数 → 后台线程批量导出
  - 抽帧模式：按数量（均匀间隔）/ 按间隔（每隔N帧）
  - 模糊帧自动过滤（拉普拉斯方差法）
  - 自动裁剪为正方形（居中裁剪/缩放填充）
  - 批量导出进度显示：总进度 + 当前视频进度 + 结果汇总
单文件，依赖：opencv-python + pillow + numpy
"""
import sys
import shutil
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk
import cv2
import numpy as np

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

APP_TITLE = "视频抽帧工具 v3（批量版）"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}


def compute_blur_score(frame: np.ndarray) -> float:
    """拉普拉斯方差法计算模糊度，分数越低越模糊"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def get_video_info(path: str) -> dict | None:
    """获取视频元信息，失败返回None"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    info = {
        "path": path,
        "name": Path(path).name,
        "stem": Path(path).stem,
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    if info["fps"] <= 0 or not np.isfinite(info["fps"]):
        info["fps"] = 30.0
    info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


class FrameExtractorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1280x820")
        self.root.minsize(1024, 700)

        # 单视频状态
        self.video_path: str | None = None
        self.cap: cv2.VideoCapture | None = None
        self._tmp_path: Path | None = None
        self.total_frames = 0
        self.fps = 0.0
        self.duration = 0.0
        self.width = 0
        self.height = 0
        self.photo: ImageTk.PhotoImage | None = None

        # 批量视频列表
        self.video_list: list[dict] = []
        self.selected_video_idx: int | None = None

        # 抽帧模式
        self.extract_mode = tk.StringVar(value="count")

        # 批量导出状态
        self.batch_running = False
        self.batch_stop_flag = False
        self.batch_thread: threading.Thread | None = None

        self._build_ui()

    # ── UI 构建 ──
    def _build_ui(self) -> None:
        # 顶部：导入按钮 + 视频信息
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=6)

        ttk.Button(top, text="导入视频", command=self._on_import).pack(side="left", padx=2)
        ttk.Button(top, text="批量导入视频", command=self._on_batch_import).pack(side="left", padx=2)
        ttk.Button(top, text="导入文件夹", command=self._on_import_folder).pack(side="left", padx=2)
        ttk.Button(top, text="清空列表", command=self._on_clear_list).pack(side="left", padx=2)

        self.info_var = tk.StringVar(value="未导入视频")
        ttk.Label(top, textvariable=self.info_var, font=("Microsoft YaHei", 10)).pack(
            side="left", padx=12)

        # 中部：左侧视频列表 + 右侧预览画布
        mid = ttk.Frame(self.root)
        mid.pack(fill="both", expand=True, padx=8, pady=4)

        # 左侧视频列表面板
        list_panel = ttk.LabelFrame(mid, text="视频列表", padding=4)
        list_panel.pack(side="left", fill="y", padx=(0, 4))

        list_header = ttk.Frame(list_panel)
        list_header.pack(fill="x", pady=(0, 2))
        ttk.Label(list_header, text="共 0 个视频", foreground="#0066cc").pack(side="left")
        self.list_count_var = tk.StringVar(value="共 0 个视频")
        ttk.Label(list_header, textvariable=self.list_count_var, foreground="#0066cc").pack(side="left")

        # Treeview 视频列表
        list_frame = ttk.Frame(list_panel)
        list_frame.pack(fill="both", expand=True)

        self.tree = ttk.Treeview(list_frame, columns=("idx", "name", "dur", "res", "frames", "status"),
                                 show="headings", height=20)
        self.tree.heading("idx", text="#")
        self.tree.heading("name", text="文件名")
        self.tree.heading("dur", text="时长")
        self.tree.heading("res", text="分辨率")
        self.tree.heading("frames", text="帧数")
        self.tree.heading("status", text="状态")
        self.tree.column("idx", width=30, anchor="center")
        self.tree.column("name", width=160, anchor="w")
        self.tree.column("dur", width=55, anchor="center")
        self.tree.column("res", width=75, anchor="center")
        self.tree.column("frames", width=55, anchor="center")
        self.tree.column("status", width=60, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        tree_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        tree_scroll.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=tree_scroll.set)

        # 右侧预览画布
        preview_panel = ttk.Frame(mid)
        preview_panel.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(preview_panel, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._show_preview())

        # 底部：抽帧参数区
        bot = ttk.Frame(self.root)
        bot.pack(fill="x", padx=8, pady=6)

        # 抽帧模式选择
        mode_frame = ttk.LabelFrame(bot, text="抽帧模式（应用于所有视频）", padding=6)
        mode_frame.pack(side="left", padx=2)

        ttk.Radiobutton(mode_frame, text="按数量", variable=self.extract_mode,
                        value="count", command=self._on_mode_change).grid(row=0, column=0, padx=4)
        ttk.Radiobutton(mode_frame, text="按间隔", variable=self.extract_mode,
                        value="interval", command=self._on_mode_change).grid(row=0, column=1, padx=4)

        # 按数量：抽取数量
        self.count_frame = ttk.Frame(mode_frame)
        self.count_frame.grid(row=1, column=0, columnspan=2, pady=4)
        ttk.Label(self.count_frame, text="每视频抽取:").pack(side="left", padx=2)
        self.count_var = tk.IntVar(value=100)
        self.count_spin = ttk.Spinbox(
            self.count_frame, from_=1, to=99999, width=8, textvariable=self.count_var)
        self.count_spin.pack(side="left", padx=2)
        ttk.Label(self.count_frame, text="张（均匀间隔）").pack(side="left", padx=2)

        # 按间隔：每隔N帧
        self.interval_frame = ttk.Frame(mode_frame)
        ttk.Label(self.interval_frame, text="每隔").pack(side="left", padx=2)
        self.interval_var = tk.IntVar(value=10)
        self.interval_spin = ttk.Spinbox(
            self.interval_frame, from_=1, to=9999, width=6, textvariable=self.interval_var)
        self.interval_spin.pack(side="left", padx=2)
        ttk.Label(self.interval_frame, text="帧抽1张").pack(side="left", padx=2)

        ttk.Separator(bot, orient="vertical").pack(side="left", fill="y", padx=8)

        # 模糊过滤
        blur_frame = ttk.LabelFrame(bot, text="模糊帧过滤", padding=6)
        blur_frame.pack(side="left", padx=2)

        self.blur_enable = tk.BooleanVar(value=False)
        ttk.Checkbutton(blur_frame, text="启用模糊帧自动过滤",
                        variable=self.blur_enable,
                        command=self._on_blur_toggle).grid(row=0, column=0, columnspan=3, sticky="w")

        ttk.Label(blur_frame, text="模糊阈值:").grid(row=1, column=0, padx=2, pady=2)
        self.blur_threshold_var = tk.IntVar(value=60)
        self.blur_spin = ttk.Spinbox(
            blur_frame, from_=1, to=500, width=6, textvariable=self.blur_threshold_var,
            state="disabled")
        self.blur_spin.grid(row=1, column=1, padx=2)
        ttk.Label(blur_frame, text="(低于此值视为模糊，显微镜建议30~80)",
                  foreground="#888").grid(row=1, column=2, padx=4)

        ttk.Separator(bot, orient="vertical").pack(side="left", fill="y", padx=8)

        # 裁剪为正方形
        crop_frame = ttk.LabelFrame(bot, text="裁剪为正方形", padding=6)
        crop_frame.pack(side="left", padx=2)

        self.crop_enable = tk.BooleanVar(value=False)
        ttk.Checkbutton(crop_frame, text="启用自动裁剪为正方形",
                        variable=self.crop_enable,
                        command=self._on_crop_toggle).grid(row=0, column=0, columnspan=4, sticky="w")

        self.crop_params_frame = ttk.Frame(crop_frame)
        self.crop_params_frame.grid(row=1, column=0, columnspan=4, pady=2)

        ttk.Label(self.crop_params_frame, text="裁剪模式:").grid(row=0, column=0, padx=2, pady=2)
        self.crop_mode = tk.StringVar(value="center")
        ttk.Radiobutton(self.crop_params_frame, text="居中裁剪", variable=self.crop_mode,
                        value="center").grid(row=0, column=1, padx=2)
        ttk.Radiobutton(self.crop_params_frame, text="缩放填充", variable=self.crop_mode,
                        value="pad").grid(row=0, column=2, padx=2)

        ttk.Label(self.crop_params_frame, text="目标尺寸:").grid(row=1, column=0, padx=2, pady=2)
        self.crop_size_var = tk.IntVar(value=640)
        self.crop_size_combo = ttk.Combobox(
            self.crop_params_frame, textvariable=self.crop_size_var, width=8,
            values=[320, 480, 640, 960, 1280, 1920])
        self.crop_size_combo.grid(row=1, column=1, padx=2)
        ttk.Label(self.crop_params_frame, text="px", foreground="#888").grid(
            row=1, column=2, padx=2, sticky="w")

        self._set_crop_controls_state("disabled")

        ttk.Separator(bot, orient="vertical").pack(side="left", fill="y", padx=8)

        # 预览帧滑块（单视频模式用）
        preview_frame = ttk.Frame(bot)
        preview_frame.pack(side="left", padx=2)
        ttk.Label(preview_frame, text="预览帧:").pack(side="left", padx=2)
        self.preview_var = tk.IntVar(value=0)
        self.preview_scale = ttk.Scale(
            preview_frame, from_=0, to=0, orient="horizontal", length=120,
            variable=self.preview_var, command=self._on_preview_change)
        self.preview_scale.pack(side="left", padx=2)
        self.preview_label = ttk.Label(preview_frame, text="0 / 0")
        self.preview_label.pack(side="left", padx=4)

        ttk.Separator(bot, orient="vertical").pack(side="left", fill="y", padx=8)

        # 导出按钮
        self.export_btn = ttk.Button(bot, text="批量导出图片", command=self._on_batch_export)
        self.export_btn.pack(side="right", padx=2)

    # ── 模式切换 ──
    def _on_mode_change(self) -> None:
        if self.extract_mode.get() == "count":
            self.interval_frame.grid_forget()
            self.count_frame.grid(row=1, column=0, columnspan=2, pady=4)
        else:
            self.count_frame.grid_forget()
            self.interval_frame.grid(row=1, column=0, columnspan=2, pady=4)

    def _on_blur_toggle(self) -> None:
        if self.blur_enable.get():
            self.blur_spin.config(state="normal")
        else:
            self.blur_spin.config(state="disabled")

    def _on_crop_toggle(self) -> None:
        state = "normal" if self.crop_enable.get() else "disabled"
        self._set_crop_controls_state(state)

    def _set_crop_controls_state(self, state: str) -> None:
        for child in self.crop_params_frame.winfo_children():
            try:
                child.config(state=state)
            except tk.TclError:
                pass

    def _crop_to_square(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        target_size = self.crop_size_var.get()
        mode = self.crop_mode.get()

        if mode == "center":
            side = min(w, h)
            x1 = (w - side) // 2
            y1 = (h - side) // 2
            cropped = frame[y1:y1 + side, x1:x1 + side]
            result = cv2.resize(cropped, (target_size, target_size),
                               interpolation=cv2.INTER_AREA if side > target_size else cv2.INTER_LINEAR)
        else:
            scale = min(target_size / w, target_size / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            result = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            x_offset = (target_size - new_w) // 2
            y_offset = (target_size - new_h) // 2
            result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return result

    # ── 视频打开/释放 ──
    def _open_video(self, path: str) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            return cap
        suffix = Path(path).suffix
        tmp = Path(tempfile.gettempdir()) / f"_frame_extract_tmp{suffix}"
        try:
            shutil.copy2(path, tmp)
        except Exception:
            return None
        cap = cv2.VideoCapture(str(tmp))
        if cap.isOpened():
            self._tmp_path = tmp
            return cap
        return None

    def _release_video(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self._tmp_path is not None and self._tmp_path.exists():
            try:
                self._tmp_path.unlink()
            except Exception:
                pass
            self._tmp_path = None

    # ── 单视频导入 ──
    def _on_import(self) -> None:
        path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", " ".join(f"*{e}" for e in sorted(VIDEO_EXTS))),
                       ("所有文件", "*.*")],
        )
        if not path:
            return
        self._load_single_video(path)

    def _load_single_video(self, path: str) -> None:
        self._release_video()
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
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        total_sec = int(self.duration)
        h, rem = divmod(total_sec, 3600)
        m, s = divmod(rem, 60)
        dur_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"

        self.info_var.set(
            f"{Path(path).name}  |  {self.width}×{self.height}  |  "
            f"{self.total_frames}帧  |  {self.fps:.1f}fps  |  时长 {dur_str}")

        max_frame = max(0, self.total_frames - 1)
        self.preview_scale.config(to=max_frame)
        self.preview_var.set(0)
        self.count_spin.config(to=max(1, self.total_frames))
        if self.count_var.get() > self.total_frames:
            self.count_var.set(self.total_frames)

        self._update_preview_label()
        self._show_preview()

    # ── 批量导入 ──
    def _on_batch_import(self) -> None:
        paths = filedialog.askopenfilenames(
            title="选择多个视频文件（可Ctrl+A全选）",
            filetypes=[("视频文件", " ".join(f"*{e}" for e in sorted(VIDEO_EXTS))),
                       ("所有文件", "*.*")],
        )
        if not paths:
            return
        self._add_videos_to_list(list(paths))

    def _on_import_folder(self) -> None:
        folder = filedialog.askdirectory(title="选择包含视频的文件夹（自动扫描所有视频）")
        if not folder:
            return
        folder_path = Path(folder)
        paths = []
        for ext in sorted(VIDEO_EXTS):
            paths.extend(folder_path.rglob(f"*{ext}"))
        if not paths:
            messagebox.showinfo(APP_TITLE, f"文件夹中未找到视频文件:\n{folder}")
            return
        self._add_videos_to_list([str(p) for p in paths])

    def _add_videos_to_list(self, paths: list[str]) -> None:
        added = 0
        skipped = 0
        for path in paths:
            # 去重
            if any(v["path"] == path for v in self.video_list):
                skipped += 1
                continue
            info = get_video_info(path)
            if info is None:
                skipped += 1
                continue
            info["status"] = "待处理"
            self.video_list.append(info)
            added += 1

        self._refresh_tree()
        msg = f"已添加 {added} 个视频"
        if skipped > 0:
            msg += f"，跳过 {skipped} 个（重复或无法打开）"
        self.info_var.set(msg)
        # 自动选中第一个
        if self.video_list and self.selected_video_idx is None:
            self.tree.selection_set(self.tree.get_children()[0])
            self._on_tree_select(None)

    def _refresh_tree(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, v in enumerate(self.video_list):
            total_sec = int(v["duration"])
            m, s = divmod(total_sec, 60)
            dur_str = f"{m:02d}:{s:02d}"
            self.tree.insert("", "end", iid=str(i), values=(
                i + 1, v["name"], dur_str,
                f"{v['width']}×{v['height']}", v["total_frames"], v["status"]))
        self.list_count_var.set(f"共 {len(self.video_list)} 个视频")

    def _on_tree_select(self, _event) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if idx >= len(self.video_list):
            return
        self.selected_video_idx = idx
        video = self.video_list[idx]
        # 加载到预览
        self._load_single_video(video["path"])

    def _on_clear_list(self) -> None:
        if self.batch_running:
            messagebox.showinfo(APP_TITLE, "批量导出中，无法清空列表")
            return
        if not self.video_list:
            return
        if not messagebox.askyesno(APP_TITLE, f"确定清空视频列表？（共 {len(self.video_list)} 个视频）"):
            return
        self.video_list.clear()
        self.selected_video_idx = None
        self._refresh_tree()
        self._release_video()
        self.info_var.set("已清空列表")
        self.preview_scale.config(to=0)
        self.preview_label.config(text="0 / 0")
        self.canvas.delete("all")

    # ── 预览 ──
    def _on_preview_change(self, _val: str) -> None:
        self._update_preview_label()
        self._show_preview()

    def _update_preview_label(self) -> None:
        if self.total_frames > 0:
            cur = self.preview_var.get()
            self.preview_label.config(text=f"{cur} / {self.total_frames - 1}")

    def _show_preview(self) -> None:
        if self.cap is None or self.total_frames == 0:
            return
        frame_idx = self.preview_var.get()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        h, w = frame.shape[:2]
        scale = min(cw / w, ch / h, 1.0)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        small = cv2.resize(frame, (nw, nh),
                           interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.canvas.delete("all")
        ix = (cw - nw) // 2
        iy = (ch - nh) // 2
        self.canvas.create_image(ix, iy, anchor="nw", image=self.photo)

    # ── 生成候选帧索引 ──
    def _generate_candidate_indices(self, total_frames: int) -> list[int]:
        if self.extract_mode.get() == "count":
            count = min(self.count_var.get(), total_frames)
            indices = np.linspace(0, total_frames - 1, count, dtype=int)
            return indices.tolist()
        else:
            interval = max(1, self.interval_var.get())
            return list(range(0, total_frames, interval))

    # ── 批量导出 ──
    def _on_batch_export(self) -> None:
        if self.batch_running:
            # 停止
            self.batch_stop_flag = True
            self.export_btn.config(text="正在停止...")
            return

        if not self.video_list:
            messagebox.showwarning(APP_TITLE, "请先导入视频（批量导入或导入文件夹）")
            return

        out_dir = filedialog.askdirectory(title="选择批量导出目录")
        if not out_dir:
            return

        # 预估总帧数
        total_candidates = 0
        for v in self.video_list:
            total_candidates += len(self._generate_candidate_indices(v["total_frames"]))

        if not messagebox.askyesno(APP_TITLE,
                                    f"即将批量导出 {len(self.video_list)} 个视频，\n"
                                    f"预计抽取约 {total_candidates} 帧，\n"
                                    f"导出目录: {out_dir}\n\n"
                                    f"是否开始？"):
            return

        self.batch_running = True
        self.batch_stop_flag = False
        self.export_btn.config(text="■ 停止导出")

        # 启动后台线程
        self.batch_thread = threading.Thread(
            target=self._batch_export_worker,
            args=(out_dir,),
            daemon=True)
        self.batch_thread.start()

        # 显示进度窗口
        self._show_batch_progress(out_dir, total_candidates)

    def _show_batch_progress(self, out_dir: str, total_candidates: int) -> None:
        self.prog_win = tk.Toplevel(self.root)
        self.prog_win.title("批量导出中...")
        self.prog_win.geometry("560x280")
        self.prog_win.transient(self.root)
        self.prog_win.grab_set()
        self.prog_win.protocol("WM_DELETE_WINDOW", lambda: None)  # 禁止关闭

        mode_text = "按数量" if self.extract_mode.get() == "count" else "按间隔"
        blur_text = f"，模糊过滤(阈值{self.blur_threshold_var.get()})" if self.blur_enable.get() else ""
        crop_text = f"，裁剪{self.crop_size_var.get()}x{self.crop_size_var.get()}" if self.crop_enable.get() else ""
        ttk.Label(self.prog_win, text=f"批量导出中（{mode_text}{blur_text}{crop_text}）",
                  font=("Microsoft YaHei", 10, "bold")).pack(pady=6)

        # 总进度
        ttk.Label(self.prog_win, text="总进度:").pack(anchor="w", padx=20)
        self.total_prog_bar = ttk.Progressbar(self.prog_win, length=500, mode="determinate",
                                                maximum=len(self.video_list))
        self.total_prog_bar.pack(pady=2, padx=20)
        self.total_prog_label = ttk.Label(self.prog_win, text=f"0 / {len(self.video_list)} 个视频")
        self.total_prog_label.pack(anchor="w", padx=20)

        # 当前视频进度
        ttk.Label(self.prog_win, text="当前视频:").pack(anchor="w", padx=20, pady=(8, 0))
        self.cur_video_label = ttk.Label(self.prog_win, text="", foreground="#0066cc")
        self.cur_video_label.pack(anchor="w", padx=20)
        self.cur_prog_bar = ttk.Progressbar(self.prog_win, length=500, mode="determinate",
                                              maximum=total_candidates)
        self.cur_prog_bar.pack(pady=2, padx=20)
        self.cur_prog_label = ttk.Label(self.prog_win, text="")
        self.cur_prog_label.pack(anchor="w", padx=20)

        # 详情
        self.detail_label = ttk.Label(self.prog_win, text="", foreground="#008800")
        self.detail_label.pack(anchor="w", padx=20, pady=(8, 0))

        # 定时检查进度
        self._check_batch_progress()

    def _check_batch_progress(self) -> None:
        if not self.batch_running:
            # 导出完成，显示结果
            self._show_batch_result()
            return

        # 更新进度（从共享状态读取）
        if hasattr(self, "_batch_state"):
            state = self._batch_state
            self.total_prog_bar["value"] = state["video_idx"]
            self.total_prog_label.config(text=f"{state['video_idx']} / {len(self.video_list)} 个视频")
            if state["current_video"]:
                self.cur_video_label.config(text=state["current_video"])
            self.cur_prog_bar["maximum"] = state["current_total"]
            self.cur_prog_bar["value"] = state["current_done"]
            self.cur_prog_label.config(
                text=f"{state['current_done']} / {state['current_total']} 帧")
            self.detail_label.config(
                text=f"已导出 {state['total_exported']} 张，模糊过滤 {state['total_blur']} 张，失败 {state['total_failed']} 张")

        self.root.after(100, self._check_batch_progress)

    def _batch_export_worker(self, out_dir: str) -> None:
        """后台线程：批量导出所有视频"""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        use_blur_filter = self.blur_enable.get()
        blur_threshold = self.blur_threshold_var.get()
        use_crop = self.crop_enable.get()

        # 共享状态
        self._batch_state = {
            "video_idx": 0,
            "current_video": "",
            "current_total": 0,
            "current_done": 0,
            "total_exported": 0,
            "total_blur": 0,
            "total_failed": 0,
        }

        results = []  # 每个视频的结果

        for i, video in enumerate(self.video_list):
            if self.batch_stop_flag:
                break

            state = self._batch_state
            state["video_idx"] = i
            state["current_video"] = video["name"]
            state["current_total"] = 0
            state["current_done"] = 0

            # 更新列表状态
            self.root.after(0, lambda idx=i: self._update_tree_status(idx, "处理中"))

            # 打开视频
            cap = cv2.VideoCapture(video["path"])
            if not cap.isOpened():
                state["total_failed"] += 1
                results.append({"name": video["name"], "exported": 0, "blur": 0, "failed": 1})
                self.root.after(0, lambda idx=i: self._update_tree_status(idx, "失败"))
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            candidates = self._generate_candidate_indices(total_frames)
            state["current_total"] = len(candidates)

            exported = 0
            filtered_blur = 0
            failed = 0

            for j, idx in enumerate(candidates):
                if self.batch_stop_flag:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret or frame is None:
                    failed += 1
                else:
                    # 模糊帧过滤
                    if use_blur_filter:
                        score = compute_blur_score(frame)
                        if score < blur_threshold:
                            filtered_blur += 1
                            state["current_done"] = j + 1
                            continue

                    # 裁剪为正方形
                    if use_crop:
                        frame = self._crop_to_square(frame)

                    filename = f"{video['stem']}_f{int(idx):04d}.png"
                    img_path = out_path / filename
                    ok, buf = cv2.imencode(".png", frame)
                    if ok:
                        buf.tofile(str(img_path))
                        exported += 1
                    else:
                        failed += 1

                state["current_done"] = j + 1

            cap.release()

            # 正确累加统计
            state["total_exported"] += exported
            state["total_blur"] += filtered_blur
            state["total_failed"] += failed

            results.append({"name": video["name"], "exported": exported,
                           "blur": filtered_blur, "failed": failed})

            status = f"导出{exported}" if failed == 0 else f"导出{exported}(失败{failed})"
            self.root.after(0, lambda idx=i, s=status: self._update_tree_status(idx, s))

        state["video_idx"] = len(self.video_list) if not self.batch_stop_flag else state["video_idx"]
        self.batch_running = False
        self._batch_results = results
        self._batch_out_dir = out_dir
        self._batch_stopped = self.batch_stop_flag

    def _update_tree_status(self, idx: int, status: str) -> None:
        if idx < len(self.video_list):
            self.video_list[idx]["status"] = status
            self.tree.set(str(idx), "status", status)

    def _show_batch_result(self) -> None:
        if hasattr(self, "prog_win") and self.prog_win.winfo_exists():
            self.prog_win.destroy()

        self.export_btn.config(text="批量导出图片")

        results = getattr(self, "_batch_results", [])
        out_dir = getattr(self, "_batch_out_dir", "")
        stopped = getattr(self, "_batch_stopped", False)

        total_exported = sum(r["exported"] for r in results)
        total_blur = sum(r["blur"] for r in results)
        total_failed = sum(r["failed"] for r in results)

        msg_lines = [
            f"批量导出{'已停止' if stopped else '完成'}！",
            f"处理视频: {len(results)} / {len(self.video_list)} 个",
            f"成功导出: {total_exported} 张",
        ]
        if self.blur_enable.get():
            msg_lines.append(f"模糊过滤: {total_blur} 张")
        if self.crop_enable.get():
            crop_mode_text = "居中裁剪" if self.crop_mode.get() == "center" else "缩放填充"
            msg_lines.append(f"裁剪为正方形: {crop_mode_text} {self.crop_size_var.get()}x{self.crop_size_var.get()}")
        if total_failed > 0:
            msg_lines.append(f"失败: {total_failed} 张")
        msg_lines.append(f"目录: {out_dir}")

        # 各视频明细
        msg_lines.append("\n── 各视频明细 ──")
        for r in results:
            line = f"  {r['name']}: 导出{r['exported']}张"
            if r['blur'] > 0:
                line += f"，模糊过滤{r['blur']}张"
            if r['failed'] > 0:
                line += f"，失败{r['failed']}张"
            msg_lines.append(line)

        messagebox.showinfo(APP_TITLE, "\n".join(msg_lines))
        print(f"[BATCH] 导出 {total_exported} 张到 {out_dir}，"
              f"模糊过滤 {total_blur} 张，失败 {total_failed} 张")

    def __del__(self):
        self._release_video()


# ── 入口 ──
def main() -> int:
    root = tk.Tk()
    FrameExtractorApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
