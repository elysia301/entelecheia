from __future__ import annotations

import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk


# ── 配置 ──────────────────────────────────────────────────────────────
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
FIXED_BOX_W = 0.015625
FIXED_BOX_H = 0.015625
# 点击命中半径（显示像素）：点中已有框 → 删除；点空白 → 新增
HIT_RADIUS_PX = 10

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")


# ── 工具函数 ──────────────────────────────────────────────────────────
def natural_sort_key(path_str: str) -> list:
    name = Path(path_str).name.lower()
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", name)]


def collect_images(input_dir: Path) -> list[Path]:
    images = [f for f in input_dir.iterdir()
              if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    images.sort(key=lambda p: natural_sort_key(str(p)))
    return images


def read_image_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return img


def parse_label_file(label_path: Path) -> list[tuple[int, float, float]]:
    """解析YOLO标签 → [(cls, xc, yc)]，宽高固定，忽略异常行"""
    boxes: list[tuple[int, float, float]] = []
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
        except (ValueError, IndexError):
            continue
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
            continue
        boxes.append((cls, xc, yc))
    return boxes


def boxes_to_text(boxes: list[tuple[int, float, float]]) -> str:
    lines = [f"{c} {x:.6f} {y:.6f} {FIXED_BOX_W:.6f} {FIXED_BOX_H:.6f}"
             for c, x, y in boxes]
    return "\n".join(lines) + ("\n" if lines else "")


# ── 编辑器主界面 ──────────────────────────────────────────────────────
class LabelEditor:
    def __init__(self, root: tk.Tk, img_dir: Path, label_dir: Path) -> None:
        self.root = root
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.images = collect_images(img_dir)
        self.idx = 0

        self.img: np.ndarray | None = None      # 原图 BGR
        self.boxes: list[tuple[int, float, float]] = []
        self.dirty = False

        self.scale = 1.0
        self.off_x = 0
        self.off_y = 0
        self._photo_ref: ImageTk.PhotoImage | None = None

        root.title("标签审核编辑器")
        root.geometry("860x860")

        # ── 顶部工具栏 ──
        bar = tk.Frame(root, pady=6)
        bar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(bar, text="⬅ 上一张", width=10,
                  command=self.prev_image).pack(side=tk.LEFT, padx=6)
        tk.Button(bar, text="下一张 ➡", width=10,
                  command=self.next_image).pack(side=tk.LEFT, padx=6)
        self.nav_label = tk.Label(bar, text="", font=("Microsoft YaHei", 11))
        self.nav_label.pack(side=tk.LEFT, padx=14)
        tk.Button(bar, text="💾 导出覆写", width=12, bg="#d0ffd0",
                  command=self.export).pack(side=tk.RIGHT, padx=6)
        self.count_label = tk.Label(bar, text="", font=("Microsoft YaHei", 11))
        self.count_label.pack(side=tk.RIGHT, padx=14)

        # ── 画布 ──
        self.canvas = tk.Canvas(root, bg="#202020", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        # ── 底部提示 ──
        hint = ("操作：左键点空白处 = 新增框 ｜ 左键点中已有框 = 删除该框 ｜ "
                "←/→ 键切换上一张/下一张 ｜ 导出覆写当前标签文件")
        tk.Label(root, text=hint, font=("Microsoft YaHei", 10),
                 fg="#555", pady=4).pack(side=tk.BOTTOM, fill=tk.X)

        root.bind("<Left>", lambda e: self.prev_image())
        root.bind("<Right>", lambda e: self.next_image())

        self.load_index(0)

    # ── 图像加载 ──
    def label_path_of(self, img_path: Path) -> Path:
        return self.label_dir / f"{img_path.stem}.txt"

    def load_index(self, i: int) -> None:
        if not self.images:
            self.nav_label.config(text="目录中没有图片")
            self.img = None
            self.boxes = []
            self.redraw()
            return
        self.idx = max(0, min(i, len(self.images) - 1))
        img_path = self.images[self.idx]
        try:
            self.img = read_image_bgr(img_path)
        except Exception as e:
            messagebox.showerror("读取失败", str(e))
            return
        self.boxes = parse_label_file(self.label_path_of(img_path))
        self.dirty = False
        self.update_status()
        self.redraw()

    def update_status(self) -> None:
        img_path = self.images[self.idx]
        label_path = self.label_path_of(img_path)
        exists = "✅" if label_path.exists() else "（无标签文件，将新建）"
        self.nav_label.config(
            text=f"[{self.idx + 1}/{len(self.images)}] {img_path.name}  {exists}")
        mark = " ●未保存" if self.dirty else ""
        self.count_label.config(text=f"框数: {len(self.boxes)}{mark}")
        self.root.title(f"标签审核编辑器 - {img_path.name}{mark}")

    # ── 绘制（修复偏移核心部分） ──
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
        # 修复：无论放大缩小都强制resize，保证图片尺寸与框计算尺寸一致
        if self.scale < 1.0:
            disp = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_AREA)
        else:
            disp = cv2.resize(rgb, (dw, dh), interpolation=cv2.INTER_LINEAR)

        self._photo_ref = ImageTk.PhotoImage(Image.fromarray(disp))
        self.canvas.create_image(self.off_x, self.off_y, anchor=tk.NW,
                                 image=self._photo_ref)

        # 红框坐标计算，与图片尺寸严格对应
        box_w = FIXED_BOX_W * w * self.scale
        box_h = FIXED_BOX_H * h * self.scale
        for _, xc, yc in self.boxes:
            cx = self.off_x + xc * dw
            cy = self.off_y + yc * dh
            x1, y1 = cx - box_w / 2, cy - box_h / 2
            x2, y2 = cx + box_w / 2, cy + box_h / 2
            self.canvas.create_rectangle(x1, y1, x2, y2,
                                         outline="#FF2020", width=2)
            # 放大时画中心点十字，便于核对
            if self.scale > 1.5:
                self.canvas.create_line(cx - 4, cy, cx + 4, cy,
                                        fill="#FF2020", width=1)
                self.canvas.create_line(cx, cy - 4, cx, cy + 4,
                                        fill="#FF2020", width=1)

    # ── 交互：点空白新增 / 点中删除 ──
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

        # 找命中框（显示距离 < 命中半径 或 落在框内）
        hit_r = max(HIT_RADIUS_PX, FIXED_BOX_W * w * self.scale)
        best_i, best_d = -1, hit_r
        for i, (_, bx, by) in enumerate(self.boxes):
            dx = (bx - xc) * dw
            dy = (by - yc) * dh
            d = (dx * dx + dy * dy) ** 0.5
            if d < best_d:
                best_i, best_d = i, d

        if best_i >= 0:
            self.boxes.pop(best_i)
        else:
            self.boxes.append((0, xc, yc))
        self.dirty = True
        self.update_status()
        self.redraw()

    # ── 翻页 ──
    def switch_to(self, new_idx: int) -> None:
        if self.dirty:
            ans = messagebox.askyesnocancel(
                "未保存的修改",
                f"当前图片有 {len(self.boxes)} 个框尚未导出。\n\n"
                "是 = 保存并翻页\n否 = 放弃修改并翻页\n取消 = 留在本页")
            if ans is None:
                return
            if ans:
                self.export(silent=True)
        self.load_index(new_idx)

    def prev_image(self) -> None:
        self.switch_to(self.idx - 1)

    def next_image(self) -> None:
        self.switch_to(self.idx + 1)

    # ── 导出覆写 ──
    def export(self, silent: bool = False) -> None:
        if not self.images:
            return
        img_path = self.images[self.idx]
        label_path = self.label_path_of(img_path)
        try:
            self.label_dir.mkdir(parents=True, exist_ok=True)
            label_path.write_text(boxes_to_text(self.boxes), encoding="utf-8")
        except Exception as e:
            messagebox.showerror("导出失败", str(e))
            return
        self.dirty = False
        self.update_status()
        if not silent:
            messagebox.showinfo(
                "导出成功",
                f"已覆写: {label_path}\n共 {len(self.boxes)} 个框")


# ── 入口 ──────────────────────────────────────────────────────────────
def main() -> int:
    # 全程只用一个Tk实例：多实例会导致PhotoImage挂错解释器而报pyimage1错误
    root = tk.Tk()
    root.withdraw()

    print("📂 请选择图片目录...")
    img_dir = filedialog.askdirectory(title="选择图片目录")
    if not img_dir:
        print("❌ 未选择图片目录，退出。")
        return 1

    print("📂 请选择坐标(YOLO标签)目录...")
    label_dir = filedialog.askdirectory(title="选择YOLO标签目录")
    if not label_dir:
        print("❌ 未选择标签目录，退出。")
        return 1

    root.deiconify()
    LabelEditor(root, Path(img_dir), Path(label_dir))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
