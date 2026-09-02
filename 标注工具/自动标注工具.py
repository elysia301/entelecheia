# -*- coding: utf-8 -*-
"""
自动标注工具（基于 KaLos618 YOLOv8s 模型）
功能：
  1. 批量导入图片目录 + 坐标(YOLO标签)目录
  2. 已有坐标 → 红框；模型推理新增目标 → 黄框（与红框IoU≥阈值的黄框自动删除及其坐标）
  3. 无坐标文件的图片 → 仅显示推理黄框
  4. 左键点击框 = 删除该框；左键点击空白 = 手动新增红框
  5. 保存按钮 / W键 / Ctrl+S → 覆写原坐标文件（无则新建）
  6. 上一张 / 下一张：按钮 或 A/D键 或 ←/→键
  7. S键 → 取消当前修改，恢复原始状态
  8. 自动批量按钮 → 逐张默认保存推理结果并连续跳下一张，处理完自动停止
  9. 推理分辨率可在UI顶部自由切换（640/960/1280/1920）
"""
from __future__ import annotations

import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

# ── 配置 ──────────────────────────────────────────────────────────────
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
MODEL_PATH = r"C:\Users\86137\Desktop\entelecheia\Demiurge\runs\train\PoleMos600\weights\best.pt"
DEFAULT_IMGSZ = 1280   # 默认模型推理分辨率（UI中可自由切换）
IMGSZ_OPTIONS = [640, 960, 1280, 1920]  # UI下拉可选分辨率列表
IOU_THRESH = 0.1     # 推理框与已有红框IoU超过此值 → 自动删除该黄框及其坐标（视为已标注，不作为新增）
REDUP_IOU_THRESH = 0.5  # 红框与红框IoU超过此值 → 删除重复红框（NMS式去重，保留面积较大的框）
MAX_AUTO_NEW_BOXES = 40  # 自动批量安全阈值：已有坐标文件的图片，单次新增黄框≥此值 → 判定为异常误检，跳过不保存
DEFAULT_AUTO_DELAY = 200  # 自动批量模式翻页延时（毫秒），可在UI中调节
DEFAULT_CONF = 0.25   # 默认置信度阈值
HIT_RADIUS_PX = 12    # 点击命中半径（像素）

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")


# ── 工具函数 ──────────────────────────────────────────────────────────
def natural_sort_key(path_str: str) -> list:
    """自然排序：数字部分按数值比较"""
    name = Path(path_str).name.lower()
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", name)]


def collect_images(input_dir: Path) -> list[Path]:
    """收集目录下所有图片并自然排序"""
    images = [f for f in input_dir.iterdir()
              if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    images.sort(key=lambda p: natural_sort_key(str(p)))
    return images


def read_image_bgr(path: Path) -> np.ndarray:
    """读取图片（兼容中文路径），返回BGR numpy数组"""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return img


def parse_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """
    解析YOLO标签文件 → [(cls, xc, yc, w, h), ...]
    归一化坐标，忽略异常行
    """
    boxes: list[tuple[int, float, float, float, float]] = []
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
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except (ValueError, IndexError):
            continue
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0
                and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            continue
        boxes.append((cls, xc, yc, w, h))
    return boxes


def boxes_to_text(boxes: list[tuple[int, float, float, float, float]]) -> str:
    """将框列表转为YOLO标签文本"""
    lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
             for c, x, y, w, h in boxes]
    return "\n".join(lines) + ("\n" if lines else "")


def xywh_to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """中心点宽高 → 左上角右下角（归一化）"""
    xc, yc, w, h = box
    return (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)


def compute_iou(box1_xywh: tuple[float, float, float, float],
                box2_xywh: tuple[float, float, float, float]) -> float:
    """计算两个框的IoU（输入为归一化xywh）"""
    x1_1, y1_1, x2_1, y2_1 = xywh_to_xyxy(box1_xywh)
    x1_2, y1_2, x2_2, y2_2 = xywh_to_xyxy(box2_xywh)
    ix1 = max(x1_1, x1_2)
    iy1 = max(y1_1, y1_2)
    ix2 = min(x2_1, x2_2)
    iy2 = min(y2_1, y2_2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def deduplicate_gt_boxes(boxes: list[tuple[int, float, float, float, float]],
                          iou_thresh: float = 0.5
                          ) -> tuple[list[tuple[int, float, float, float, float]], int]:
    """
    红框间NMS式去重：按面积降序排列，依次保留，与已保留框IoU超阈值的删除。
    返回 (去重后的框列表, 删除的框数)
    """
    if len(boxes) <= 1:
        return list(boxes), 0
    indexed = [(i, b, b[3] * b[4]) for i, b in enumerate(boxes)]
    indexed.sort(key=lambda x: x[2], reverse=True)

    kept: list[tuple[int, float, float, float, float]] = []
    removed = 0
    for _, box, _area in indexed:
        overlap = False
        for k in kept:
            if compute_iou((box[1], box[2], box[3], box[4]),
                            (k[1], k[2], k[3], k[4])) >= iou_thresh:
                overlap = True
                break
        if overlap:
            removed += 1
        else:
            kept.append(box)
    return kept, removed


# ── 模型推理封装 ──────────────────────────────────────────────────────
class ModelInferencer:
    """YOLO模型推理封装，懒加载，imgsz可动态修改"""

    def __init__(self, model_path: str, imgsz: int = DEFAULT_IMGSZ) -> None:
        self.model_path = model_path
        self.imgsz = imgsz
        self._model = None
        self._device = "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO
        import torch
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[模型] 加载 {self.model_path} (device={self._device}) ...")
        self._model = YOLO(self.model_path)
        print(f"[模型] 加载完成，推理分辨率={self.imgsz}")

    def predict(self, img_bgr: np.ndarray, conf: float = DEFAULT_CONF
                ) -> list[tuple[int, float, float, float, float, float]]:
        """
        推理单张图片
        返回: [(cls, xc, yc, w, h, conf), ...]  归一化xywh
        """
        self._ensure_loaded()
        h, w = img_bgr.shape[:2]
        results = self._model.predict(
            source=img_bgr,
            imgsz=self.imgsz,
            conf=conf,
            iou=0.7,
            device=self._device,
            verbose=False,
        )
        det = []
        if not results or results[0].boxes is None:
            return det
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            xc = (x1 + x2) / 2 / w
            yc = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            det.append((cls, xc, yc, bw, bh, cf))
        return det


# ── 自动标注编辑器 ────────────────────────────────────────────────────
class AutoLabelEditor:
    def __init__(self, root: tk.Tk, img_dir: Path, label_dir: Path,
                 inferencer: ModelInferencer) -> None:
        self.root = root
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.inferencer = inferencer
        self.images = collect_images(img_dir)
        self.idx = 0
        self.conf_threshold = DEFAULT_CONF
        self.iou_thresh_gp = IOU_THRESH        # 黄框-红框去重IoU阈值（实例变量，可UI调节）
        self.iou_thresh_gg = REDUP_IOU_THRESH  # 红框-红框去重IoU阈值（实例变量，可UI调节）
        self.auto_delay = DEFAULT_AUTO_DELAY    # 自动批量翻页延时（毫秒，可UI调节）

        # 当前图片状态
        self.img: np.ndarray | None = None
        self.gt_boxes: list[tuple[int, float, float, float, float]] = []   # 红框：已有标注
        self.pred_boxes: list[tuple[int, float, float, float, float, float]] = []  # 黄框：模型新增(cls,xc,yc,w,h,conf)
        self.deleted_pred: set[int] = set()  # 被用户删除的黄框索引
        self.auto_removed_count: int = 0      # 因与红框IoU超阈值而被自动删除的黄框数
        self.gg_removed_count: int = 0        # 红框间去重删除的重复红框数
        self.auto_mode: bool = False           # 自动批量模式标志
        self.skipped_images: list[str] = []     # 自动模式下因异常而跳过的图片名
        self.auto_log: list[dict] = []           # 自动模式详细日志（每张图的处理记录）
        self._user_conf_backup: float = DEFAULT_CONF  # 自动模式前备份用户置信度
        self.dirty = False

        # 画布缩放参数
        self.scale = 1.0
        self.off_x = 0
        self.off_y = 0
        self._photo_ref: ImageTk.PhotoImage | None = None

        root.title("自动标注工具 - KaLos618")
        root.geometry("1100x900")

        # ── 顶部工具栏 ──
        bar = tk.Frame(root, pady=6)
        bar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(bar, text="⬅ 上一张", width=10,
                  command=self.prev_image).pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="下一张 ➡", width=10,
                  command=self.next_image).pack(side=tk.LEFT, padx=4)

        self.nav_label = tk.Label(bar, text="", font=("Microsoft YaHei", 10))
        self.nav_label.pack(side=tk.LEFT, padx=10)

        # 参数调节区：置信度 + 黄红去重IoU + 红红去重IoU + 翻页延时 + 推理分辨率 + 应用按钮
        param_frame = tk.Frame(bar)
        param_frame.pack(side=tk.LEFT, padx=(20, 2))

        tk.Label(param_frame, text="置信度:", font=("Microsoft YaHei", 10)).pack(side=tk.LEFT, padx=2)
        self.conf_var = tk.StringVar(value=f"{DEFAULT_CONF:.2f}")
        tk.Entry(param_frame, textvariable=self.conf_var, width=5).pack(side=tk.LEFT, padx=2)

        tk.Label(param_frame, text="黄红IoU:", font=("Microsoft YaHei", 10),
                 fg="#CC9900").pack(side=tk.LEFT, padx=(8, 2))
        self.iou_gp_var = tk.StringVar(value=f"{IOU_THRESH:.2f}")
        tk.Entry(param_frame, textvariable=self.iou_gp_var, width=5).pack(side=tk.LEFT, padx=2)

        tk.Label(param_frame, text="红红IoU:", font=("Microsoft YaHei", 10),
                 fg="#CC3333").pack(side=tk.LEFT, padx=(8, 2))
        self.iou_gg_var = tk.StringVar(value=f"{REDUP_IOU_THRESH:.2f}")
        tk.Entry(param_frame, textvariable=self.iou_gg_var, width=5).pack(side=tk.LEFT, padx=2)

        tk.Label(param_frame, text="延时ms:", font=("Microsoft YaHei", 10),
                 fg="#3366CC").pack(side=tk.LEFT, padx=(8, 2))
        self.auto_delay_var = tk.StringVar(value=str(DEFAULT_AUTO_DELAY))
        tk.Entry(param_frame, textvariable=self.auto_delay_var, width=5).pack(side=tk.LEFT, padx=2)

        # 推理分辨率下拉选择
        tk.Label(param_frame, text="推理分辨率:", font=("Microsoft YaHei", 10),
                 fg="#006633").pack(side=tk.LEFT, padx=(8, 2))
        self.imgsz_var = tk.StringVar(value=str(DEFAULT_IMGSZ))
        self.imgsz_combo = ttk.Combobox(
            param_frame, textvariable=self.imgsz_var, width=6, state="readonly",
            values=[str(s) for s in IMGSZ_OPTIONS])
        self.imgsz_combo.pack(side=tk.LEFT, padx=2)

        tk.Button(param_frame, text="应用", width=6, bg="#e0e0ff",
                  command=self.apply_thresholds).pack(side=tk.LEFT, padx=(8, 2))
        tk.Button(bar, text="重新推理", width=8,
                  command=self.re_infer).pack(side=tk.LEFT, padx=4)

        # 保存按钮
        tk.Button(bar, text="💾 保存 (W/Ctrl+S)", width=16, bg="#d0ffd0",
                  command=self.export).pack(side=tk.RIGHT, padx=6)
        # 自动批量按钮
        self.auto_btn = tk.Button(bar, text="▶ 自动批量", width=12, bg="#f0f0f0",
                                   command=self.toggle_auto_mode)
        self.auto_btn.pack(side=tk.RIGHT, padx=6)
        self.count_label = tk.Label(bar, text="", font=("Microsoft YaHei", 10))
        self.count_label.pack(side=tk.RIGHT, padx=10)

        # ── 画布 ──
        self.canvas = tk.Canvas(root, bg="#202020", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        # ── 底部图例 + 提示 ──
        bottom = tk.Frame(root)
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        legend = tk.Frame(bottom)
        legend.pack(side=tk.LEFT, padx=10, pady=4)
        tk.Label(legend, text="■", fg="#FF2020", font=("Arial", 14)).pack(side=tk.LEFT)
        tk.Label(legend, text="已有标注  ", font=("Microsoft YaHei", 9)).pack(side=tk.LEFT)
        tk.Label(legend, text="■", fg="#FFD000", font=("Arial", 14)).pack(side=tk.LEFT)
        tk.Label(legend, text="模型新增  ", font=("Microsoft YaHei", 9)).pack(side=tk.LEFT)

        hint = ("左键点框=删除 ｜ 左键点空白=手动新增 ｜ "
                "A/←=上一张 D/→=下一张 S=取消 W=保存 ｜ "
                f"自动批量:新增≥{MAX_AUTO_NEW_BOXES}跳过 ｜ "
                "置信度/黄红IoU/红红IoU/延时/推理分辨率顶部调节后点应用，单张与自动模式通用")
        tk.Label(bottom, text=hint, font=("Microsoft YaHei", 9),
                 fg="#555", pady=4).pack(side=tk.RIGHT, padx=10)

        # 快捷键
        root.bind("<Left>", lambda e: self.prev_image())
        root.bind("<Right>", lambda e: self.next_image())
        root.bind("<Control-s>", lambda e: self.export())
        root.bind("<Control-S>", lambda e: self.export())
        # WASD 快捷键：A上一张 D下一张 S取消 W保存
        root.bind("<a>", lambda e: self.prev_image())
        root.bind("<A>", lambda e: self.prev_image())
        root.bind("<d>", lambda e: self.next_image())
        root.bind("<D>", lambda e: self.next_image())
        root.bind("<s>", lambda e: self.cancel_changes())
        root.bind("<S>", lambda e: self.cancel_changes())
        root.bind("<w>", lambda e: self.export())
        root.bind("<W>", lambda e: self.export())

        self.load_index(0)

    # ── 路径辅助 ──
    def label_path_of(self, img_path: Path) -> Path:
        return self.label_dir / f"{img_path.stem}.txt"

    # ── 加载单张图片 ──
    def load_index(self, i: int) -> None:
        if not self.images:
            self.nav_label.config(text="目录中没有图片")
            self.img = None
            self.gt_boxes = []
            self.pred_boxes = []
            self.deleted_pred.clear()
            self.auto_removed_count = 0
            self.gg_removed_count = 0
            self.redraw()
            return

        self.idx = max(0, min(i, len(self.images) - 1))
        img_path = self.images[self.idx]

        # 读取图片
        try:
            self.img = read_image_bgr(img_path)
        except Exception as e:
            messagebox.showerror("读取失败", str(e))
            return

        # 读取已有标签（红框），并应用红框间去重
        raw_gt = parse_label_file(self.label_path_of(img_path))
        self.gt_boxes, self.gg_removed_count = deduplicate_gt_boxes(raw_gt, self.iou_thresh_gg)

        # 模型推理（黄框）
        self._run_inference()
        self.deleted_pred.clear()
        self.dirty = False
        self.update_status()
        self.redraw()

    def _run_inference(self) -> None:
        """对当前图片推理，过滤掉与已有框IoU过高的检测（视为已标注）"""
        if self.img is None:
            self.pred_boxes = []
            return
        try:
            raw = self.inferencer.predict(self.img, conf=self.conf_threshold)
        except Exception as e:
            messagebox.showerror("推理失败", str(e))
            self.pred_boxes = []
            return

        # IoU去重：与已有红框重叠超过阈值的检测框自动删除（不显示、不保存）
        filtered = []
        auto_removed = 0
        for det in raw:
            cls, xc, yc, w, h, cf = det
            max_iou = 0.0
            for gt in self.gt_boxes:
                iou = compute_iou((xc, yc, w, h), (gt[1], gt[2], gt[3], gt[4]))
                if iou > max_iou:
                    max_iou = iou
            if max_iou >= self.iou_thresh_gp:
                auto_removed += 1  # 与红框重叠超阈值 → 自动删除该黄框及其坐标
            else:
                filtered.append(det)
        self.pred_boxes = filtered
        self.auto_removed_count = auto_removed

    def re_infer(self) -> None:
        """重新推理（置信度改变后）"""
        try:
            self.conf_threshold = float(self.conf_var.get())
        except ValueError:
            messagebox.showwarning("输入错误", "置信度必须是数字")
            return
        if not (0.0 < self.conf_threshold < 1.0):
            messagebox.showwarning("输入错误", "置信度范围 0~1")
            return
        self._run_inference()
        self.deleted_pred.clear()
        self.dirty = True
        self.update_status()
        self.redraw()

    def apply_thresholds(self) -> None:
        """应用UI中的阈值设置：置信度 + 黄红去重IoU + 红红去重IoU + 翻页延时 + 推理分辨率，然后重新加载当前图"""
        # 读取并验证参数
        try:
            conf = float(self.conf_var.get())
            iou_gp = float(self.iou_gp_var.get())
            iou_gg = float(self.iou_gg_var.get())
            auto_delay = int(float(self.auto_delay_var.get()))
            imgsz = int(self.imgsz_var.get())
        except ValueError:
            messagebox.showwarning("输入错误", "置信度、IoU阈值、延时和分辨率必须是数字")
            return
        if not (0.0 < conf < 1.0):
            messagebox.showwarning("输入错误", "置信度范围 0~1")
            return
        if not (0.0 <= iou_gp <= 1.0):
            messagebox.showwarning("输入错误", "黄红IoU阈值范围 0~1")
            return
        if not (0.0 <= iou_gg <= 1.0):
            messagebox.showwarning("输入错误", "红红IoU阈值范围 0~1")
            return
        if not (10 <= auto_delay <= 10000):
            messagebox.showwarning("输入错误", "翻页延时范围 10~10000 毫秒")
            return
        if imgsz not in IMGSZ_OPTIONS:
            messagebox.showwarning("输入错误", f"推理分辨率必须是 {IMGSZ_OPTIONS} 之一")
            return

        self.conf_threshold = conf
        self.iou_thresh_gp = iou_gp
        self.iou_thresh_gg = iou_gg
        self.auto_delay = auto_delay

        # 推理分辨率改变 → 更新 inferencer.imgsz
        imgsz_changed = (self.inferencer.imgsz != imgsz)
        self.inferencer.imgsz = imgsz
        if imgsz_changed:
            print(f"[配置] 推理分辨率已切换为 {imgsz}")

        # 自动模式下不允许手动改阈值（避免冲突）
        if self.auto_mode:
            messagebox.showinfo("提示", "自动模式运行中，请先停止自动批量再调节阈值")
            return

        # 重新加载当前图（重新读取标签 + 红红去重 + 推理 + 黄红去重）
        self.load_index(self.idx)

    # ── 状态栏 ──
    def update_status(self) -> None:
        img_path = self.images[self.idx]
        label_path = self.label_path_of(img_path)
        exists = "✅" if label_path.exists() else "（无标签，保存将新建）"
        active_pred = len(self.pred_boxes) - len(self.deleted_pred)
        self.nav_label.config(
            text=f"[{self.idx + 1}/{len(self.images)}] {img_path.name}  {exists}")
        mark = " ●未保存" if self.dirty else ""
        auto_info = f"  黄红去重:{self.auto_removed_count}" if self.auto_removed_count > 0 else ""
        gg_info = f"  红红去重:{self.gg_removed_count}" if self.gg_removed_count > 0 else ""
        skip_info = f"  已跳过:{len(self.skipped_images)}" if self.skipped_images else ""
        mode_info = "  [自动模式]" if self.auto_mode else ""
        imgsz_info = f"  分辨率:{self.inferencer.imgsz}"
        self.count_label.config(
            text=f"红框:{len(self.gt_boxes)}  黄框:{active_pred}/{len(self.pred_boxes)}{gg_info}{auto_info}{skip_info}{mode_info}{imgsz_info}{mark}")
        self.root.title(f"自动标注工具 - {img_path.name}{mark}")

    # ── 绘制 ──
    def redraw(self) -> None:
        self.canvas.delete("all")
        if self.img is None:
            return

        cw = max(self.canvas.winfo_width(), 10)
        ch = max(self.canvas.winfo_height(), 10)
        h, w = self.img.shape[:2]
        self.scale = min(cw / w, ch / h)
        dw, dh = int(w * self.scale), int(h * self.scale)
        self.off_x = (cw - dw) // 2
        self.off_y = (ch - dh) // 2

        rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        if self.scale < 1.0:
            disp = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_AREA)
        else:
            disp = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)

        self._photo_ref = ImageTk.PhotoImage(Image.fromarray(disp))
        self.canvas.create_image(self.off_x, self.off_y, anchor=tk.NW,
                                 image=self._photo_ref)

        # 红框：已有标注
        for cls, xc, yc, bw, bh in self.gt_boxes:
            cx = self.off_x + xc * dw
            cy = self.off_y + yc * dh
            x1, y1 = cx - bw * dw / 2, cy - bh * dh / 2
            x2, y2 = cx + bw * dw / 2, cy + bh * dh / 2
            self.canvas.create_rectangle(x1, y1, x2, y2,
                                         outline="#FF2020", width=2)

        # 黄框：模型新增（未被删除的）
        for i, (cls, xc, yc, bw, bh, cf) in enumerate(self.pred_boxes):
            if i in self.deleted_pred:
                continue
            cx = self.off_x + xc * dw
            cy = self.off_y + yc * dh
            x1, y1 = cx - bw * dw / 2, cy - bh * dh / 2
            x2, y2 = cx + bw * dw / 2, cy + bh * dh / 2
            self.canvas.create_rectangle(x1, y1, x2, y2,
                                         outline="#FFD000", width=2)
            # 置信度标签
            self.canvas.create_text(x1 + 2, y1 - 2, anchor=tk.SW,
                                    text=f"{cf:.2f}", fill="#FFD000",
                                    font=("Arial", 8))

    # ── 交互：点击删除框 / 空白新增 ──
    def on_click(self, event: tk.Event) -> None:
        if self.img is None:
            return
        h, w = self.img.shape[:2]
        dw, dh = int(w * self.scale), int(h * self.scale)

        mx = event.x - self.off_x
        my = event.y - self.off_y
        if not (0 <= mx < dw and 0 <= my < dh):
            return
        xc = mx / dw
        yc = my / dh

        # 先找命中的黄框（优先，因为黄框在上面）
        hit_pred = -1
        best_dist = HIT_RADIUS_PX
        for i, (_, px, py, pw, ph, _) in enumerate(self.pred_boxes):
            if i in self.deleted_pred:
                continue
            dx = (px - xc) * dw
            dy = (py - yc) * dh
            dist = (dx * dx + dy * dy) ** 0.5
            half_w = pw * dw / 2
            half_h = ph * dh / 2
            in_box = (abs(dx) <= half_w and abs(dy) <= half_h)
            if in_box or dist < best_dist:
                hit_pred = i
                best_dist = dist

        if hit_pred >= 0:
            self.deleted_pred.add(hit_pred)
            self.dirty = True
            self.update_status()
            self.redraw()
            return

        # 再找命中的红框
        hit_gt = -1
        best_dist = HIT_RADIUS_PX
        for i, (_, gx, gy, gw, gh) in enumerate(self.gt_boxes):
            dx = (gx - xc) * dw
            dy = (gy - yc) * dh
            dist = (dx * dx + dy * dy) ** 0.5
            half_w = gw * dw / 2
            half_h = gh * dh / 2
            in_box = (abs(dx) <= half_w and abs(dy) <= half_h)
            if in_box or dist < best_dist:
                hit_gt = i
                best_dist = dist

        if hit_gt >= 0:
            self.gt_boxes.pop(hit_gt)
            self.dirty = True
            self.update_status()
            self.redraw()
            return

        # 点空白 → 手动新增红框（用平均框大小，或固定小框）
        if self.gt_boxes:
            avg_w = float(np.mean([b[3] for b in self.gt_boxes]))
            avg_h = float(np.mean([b[4] for b in self.gt_boxes]))
        else:
            avg_w, avg_h = 0.02, 0.02
        self.gt_boxes.append((0, xc, yc, avg_w, avg_h))
        self.dirty = True
        self.update_status()
        self.redraw()

    # ── 翻页（带未保存提示）──
    def switch_to(self, new_idx: int) -> None:
        # 自动模式下直接翻页，不弹窗（由 auto_step 负责保存）
        if self.auto_mode:
            self.load_index(new_idx)
            return
        if self.dirty:
            ans = messagebox.askyesnocancel(
                "未保存的修改",
                f"当前图片有未保存修改（红框{len(self.gt_boxes)} + "
                f"黄框{len(self.pred_boxes) - len(self.deleted_pred)}）。\n\n"
                "是 = 保存并翻页\n否 = 放弃修改并翻页\n取消 = 留在本页")
            if ans is None:
                return
            if ans:
                self.export(silent=True)
        self.load_index(new_idx)

    def prev_image(self) -> None:
        # 手动翻页时停止自动模式
        if self.auto_mode:
            self.auto_mode = False
            self.update_auto_button()
        if self.idx > 0:
            self.switch_to(self.idx - 1)

    def next_image(self) -> None:
        # 手动翻页时停止自动模式
        if self.auto_mode:
            self.auto_mode = False
            self.update_auto_button()
        if self.idx < len(self.images) - 1:
            self.switch_to(self.idx + 1)

    # ── S键：取消当前修改，重新加载原始状态 ──
    def cancel_changes(self) -> None:
        """放弃当前图片所有未保存修改，恢复到加载时的原始状态"""
        if self.auto_mode:
            self.auto_mode = False
            self.update_auto_button()
        if not self.images:
            return
        if not self.dirty:
            return  # 没有修改，无需取消
        self.load_index(self.idx)  # 重新加载，丢弃所有修改

    # ── 自动批量模式 ──
    def update_auto_button(self) -> None:
        if self.auto_mode:
            self.auto_btn.config(text="⏹ 停止自动", bg="#ffd0d0")
        else:
            self.auto_btn.config(text="▶ 自动批量", bg="#f0f0f0")

    def toggle_auto_mode(self) -> None:
        """切换自动批量模式：开启后逐张保存推理结果并自动跳下一张
        置信度/IoU阈值与单张模式完全一致，均受顶部UI调节控制。"""
        if self.auto_mode:
            self.auto_mode = False
            self.update_auto_button()
            return
        if not self.images:
            return
        self.skipped_images.clear()
        self.auto_log.clear()
        self.auto_mode = True
        self.update_auto_button()
        self.update_status()
        self.redraw()
        self.auto_step()

    def auto_step(self) -> None:
        """自动模式单步：异常检测 → 保存/跳过 → 记录日志 → 跳下一张 → 调度下一步"""
        if not self.auto_mode or not self.images:
            self.auto_mode = False
            self.update_auto_button()
            return

        img_path = self.images[self.idx]
        label_path = self.label_path_of(img_path)
        gt_count = len(self.gt_boxes)
        active_pred = len(self.pred_boxes) - len(self.deleted_pred)
        has_gt_file = label_path.exists()

        # ── 安全阈值检测 ──
        if has_gt_file and active_pred >= MAX_AUTO_NEW_BOXES:
            skip_reason = f"新增{active_pred}个≥阈值{MAX_AUTO_NEW_BOXES}"
            self.skipped_images.append(f"{img_path.name} ({skip_reason})")
            self.dirty = False  # 不保存，丢弃本次推理修改
            self.auto_log.append({
                'idx': self.idx + 1,
                'name': img_path.name,
                'action': '跳过',
                'gt_count': gt_count,
                'new_count': active_pred,
                'total_count': gt_count,
                'reason': skip_reason,
            })
        else:
            self.export(silent=True)
            total_after = len(self.gt_boxes)
            self.auto_log.append({
                'idx': self.idx + 1,
                'name': img_path.name,
                'action': '保存',
                'gt_count': gt_count,
                'new_count': active_pred,
                'total_count': total_after,
                'reason': '新建文件' if not has_gt_file else '覆写文件',
            })

        self.update_status()

        if self.idx >= len(self.images) - 1:
            self.auto_mode = False
            self.update_auto_button()
            self.show_auto_log()
            return

        self.idx += 1
        self.load_index(self.idx)

        self.root.after(self.auto_delay, self.auto_step)

    def show_auto_log(self) -> None:
        """弹出带滚动条的详细日志窗口，展示自动批量每张图的处理记录"""
        saved = [l for l in self.auto_log if l['action'] == '保存']
        skipped = [l for l in self.auto_log if l['action'] == '跳过']
        total_new = sum(l['new_count'] for l in saved)
        total_boxes = sum(l['total_count'] for l in saved)

        lines = []
        lines.append("=" * 70)
        lines.append("  自动批量处理日志")
        lines.append("=" * 70)
        lines.append(f"  处理总数: {len(self.auto_log)} 张")
        lines.append(f"  成功保存: {len(saved)} 张  |  跳过: {len(skipped)} 张")
        lines.append(f"  新增坐标: {total_new} 个  |  保存后总坐标: {total_boxes} 个")
        lines.append(f"  置信度: {self.conf_threshold:.2f}  |  黄红IoU: {self.iou_thresh_gp:.2f}  |  红红IoU: {self.iou_thresh_gg:.2f}")
        lines.append(f"  推理分辨率: {self.inferencer.imgsz}  |  翻页延时: {self.auto_delay}ms  |  新增阈值: {MAX_AUTO_NEW_BOXES}")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"{'序号':<5}{'文件名':<35}{'操作':<6}{'原有':<6}{'新增':<6}{'总数':<6}{'备注'}")
        lines.append("-" * 90)
        for log in self.auto_log:
            name = log['name']
            if len(name) > 33:
                name = name[:30] + "..."
            lines.append(
                f"{log['idx']:<5}{name:<35}{log['action']:<6}"
                f"{log['gt_count']:<6}{log['new_count']:<6}{log['total_count']:<6}{log['reason']}")
        lines.append("")
        lines.append("=" * 70)
        if skipped:
            lines.append("  ⚠ 跳过清单（请手动检查这些图片）：")
            for s in skipped:
                lines.append(f"    [{s['idx']}] {s['name']} - {s['reason']}")
        else:
            lines.append("  ✓ 无异常跳过，全部保存成功")
        lines.append("=" * 70)

        log_text = "\n".join(lines)

        win = tk.Toplevel(self.root)
        win.title("自动批量处理日志")
        win.geometry("900x650")
        win.transient(self.root)

        summary = (f"处理 {len(self.auto_log)} 张 | 保存 {len(saved)} 张 | "
                   f"跳过 {len(skipped)} 张 | 新增 {total_new} 坐标")
        tk.Label(win, text=summary, font=("Microsoft YaHei", 11, "bold"),
                 pady=8).pack(side=tk.TOP, fill=tk.X)

        text_frame = tk.Frame(win)
        text_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget = tk.Text(text_frame, wrap=tk.NONE, font=("Consolas", 10),
                              yscrollcommand=scrollbar.set, bg="#1e1e1e", fg="#d4d4d4")
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        text_widget.insert(tk.END, log_text)
        text_widget.config(state=tk.DISABLED)

        btn_frame = tk.Frame(win, pady=8)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)

        def save_log_to_file():
            save_path = filedialog.asksaveasfilename(
                title="保存日志",
                defaultextension=".txt",
                initialfile="auto_label_log.txt",
                filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")])
            if save_path:
                try:
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(log_text)
                    messagebox.showinfo("保存成功", f"日志已保存到:\n{save_path}")
                except Exception as e:
                    messagebox.showerror("保存失败", str(e))

        tk.Button(btn_frame, text="💾 导出日志", width=12,
                  command=save_log_to_file).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="关闭", width=10,
                  command=win.destroy).pack(side=tk.RIGHT, padx=10)

    # ── 保存（覆写原坐标文件 / 新建）──
    def export(self, silent: bool = False) -> None:
        if not self.images:
            return
        img_path = self.images[self.idx]
        label_path = self.label_path_of(img_path)

        gt_count = len(self.gt_boxes)
        active_pred = len(self.pred_boxes) - len(self.deleted_pred)

        merged: list[tuple[int, float, float, float, float]] = list(self.gt_boxes)
        for i, det in enumerate(self.pred_boxes):
            if i in self.deleted_pred:
                continue
            cls, xc, yc, w, h, _cf = det
            merged.append((cls, xc, yc, w, h))

        try:
            self.label_dir.mkdir(parents=True, exist_ok=True)
            label_path.write_text(boxes_to_text(merged), encoding="utf-8")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))
            return

        self.dirty = False
        self.gt_boxes = merged
        self.pred_boxes = []
        self.deleted_pred.clear()
        self.update_status()
        self.redraw()

        if not silent:
            is_new = "新建" if not label_path.exists() else "覆写"
            messagebox.showinfo(
                "保存成功",
                f"已{is_new}: {label_path}\n共 {len(merged)} 个框\n"
                f"(原有标注 {gt_count} + 模型新增 {active_pred})")


# ── 入口 ──────────────────────────────────────────────────────────────
def main() -> int:
    root = tk.Tk()
    root.withdraw()

    print("📂 请选择图片目录...")
    img_dir = filedialog.askdirectory(title="选择图片目录")
    if not img_dir:
        print("❌ 未选择图片目录，退出。")
        return 1

    print("📂 请选择坐标(YOLO标签)目录...")
    label_dir = filedialog.askdirectory(title="选择YOLO标签目录（可空，保存时自动新建）")
    if not label_dir:
        print("❌ 未选择标签目录，退出。")
        return 1

    if not Path(MODEL_PATH).exists():
        messagebox.showerror("模型缺失", f"未找到模型文件:\n{MODEL_PATH}")
        return 1

    inferencer = ModelInferencer(MODEL_PATH, DEFAULT_IMGSZ)

    root.deiconify()
    AutoLabelEditor(root, Path(img_dir), Path(label_dir), inferencer)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
