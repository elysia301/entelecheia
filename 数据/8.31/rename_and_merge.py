# -*- coding: utf-8 -*-
"""
将8.31数据按dataset命名规范改名并加入dataset
- 新前缀: PoleMos600
- 命名: PoleMos600_f{帧号}.png/.txt
- 分配: 帧号%10==0 → val，其余 → train（约90:10，与现有dataset一致）
"""
import sys
import re
import shutil
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC_IMG = Path(r"C:\Users\86137\Desktop\entelecheia\数据\8.31\images")
SRC_LBL = Path(r"C:\Users\86137\Desktop\entelecheia\数据\8.31\labels")
DST_BASE = Path(r"C:\Users\86137\Desktop\entelecheia\Demiurge\datasets\dataset")
NEW_PREFIX = "PoleMos600"
VAL_INTERVAL = 10  # 帧号能被此值整除 → val

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def extract_frame_num(filename):
    """从文件名提取帧号，如 'xxx_f0012.png' → 12"""
    m = re.search(r'f(\d{4})', filename)
    if m:
        return int(m.group(1))
    return None


def main():
    print("=" * 70)
    print("  8.31数据改名并加入dataset")
    print("=" * 70)
    print(f"源图片目录: {SRC_IMG}")
    print(f"源标签目录: {SRC_LBL}")
    print(f"目标目录: {DST_BASE}")
    print(f"新前缀: {NEW_PREFIX}")
    print(f"val分配规则: 帧号 % {VAL_INTERVAL} == 0")

    # 收集源图片
    src_images = [f for f in SRC_IMG.iterdir()
                  if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
    src_images.sort(key=lambda p: p.name)
    print(f"\n源图片数: {len(src_images)}")

    # 预检查：对应标签是否存在，目标是否已存在
    missing_labels = []
    existing_targets = []
    plan = []  # (src_img, src_lbl, dst_img, dst_lbl, split)

    for img_path in src_images:
        frame = extract_frame_num(img_path.name)
        if frame is None:
            print(f"  ⚠ 无法提取帧号: {img_path.name}，跳过")
            continue

        new_stem = f"{NEW_PREFIX}_f{frame:04d}"
        split = "val" if frame % VAL_INTERVAL == 0 else "train"

        src_lbl = SRC_LBL / f"{img_path.stem}.txt"
        dst_img = DST_BASE / "images" / split / f"{new_stem}{img_path.suffix}"
        dst_lbl = DST_BASE / "labels" / split / f"{new_stem}.txt"

        if not src_lbl.exists():
            missing_labels.append(img_path.name)
        if dst_img.exists() or dst_lbl.exists():
            existing_targets.append(new_stem)

        plan.append((img_path, src_lbl, dst_img, dst_lbl, split, frame))

    if missing_labels:
        print(f"\n❌ 缺少标签的图片: {len(missing_labels)} 个")
        for n in missing_labels[:5]:
            print(f"    {n}")
        print("  终止执行，请先补齐标签")
        return 1

    if existing_targets:
        print(f"\n⚠ 目标已存在的文件: {len(existing_targets)} 个")
        for n in existing_targets[:5]:
            print(f"    {n}")
        print("  终止执行，避免覆盖")
        return 1

    print(f"\n预检查通过: 标签齐全，无目标冲突")

    # 统计分配
    train_count = sum(1 for p in plan if p[4] == "train")
    val_count = sum(1 for p in plan if p[4] == "val")
    print(f"计划分配: train={train_count}, val={val_count}")

    # 执行复制
    print(f"\n开始复制...")
    success = 0
    failed = 0

    for i, (src_img, src_lbl, dst_img, dst_lbl, split, frame) in enumerate(plan, 1):
        try:
            # 确保目标目录存在
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)

            # 复制图片和标签
            shutil.copy2(str(src_img), str(dst_img))
            shutil.copy2(str(src_lbl), str(dst_lbl))
            success += 1

            if i % 100 == 0 or i == len(plan):
                print(f"  进度: {i}/{len(plan)} ({i*100//len(plan)}%)")
        except Exception as e:
            print(f"  ❌ 复制失败: {src_img.name} → {e}")
            failed += 1

    print(f"\n复制完成: 成功={success}, 失败={failed}")

    # 验证结果
    print(f"\n=== 验证结果 ===")
    for split in ["train", "val"]:
        dst_imgs = list((DST_BASE / "images" / split).glob(f"{NEW_PREFIX}_*"))
        dst_lbls = list((DST_BASE / "labels" / split).glob(f"{NEW_PREFIX}_*"))
        print(f"  {split}: 新增图片={len(dst_imgs)}, 新增标签={len(dst_lbls)}")
        if dst_imgs:
            print(f"    示例: {dst_imgs[0].name} ~ {dst_imgs[-1].name}")

    # 全局统计
    total_train_img = len(list((DST_BASE / "images" / "train").iterdir()))
    total_train_lbl = len(list((DST_BASE / "labels" / "train").iterdir()))
    total_val_img = len(list((DST_BASE / "images" / "val").iterdir()))
    total_val_lbl = len(list((DST_BASE / "labels" / "val").iterdir()))
    print(f"\ndataset 总计: train={total_train_img}图/{total_train_lbl}标, "
          f"val={total_val_img}图/{total_val_lbl}标")

    if failed == 0 and success == len(plan):
        print(f"\n✅ 全部成功，共加入 {success} 对文件")
    else:
        print(f"\n⚠ 部分失败，请检查")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
