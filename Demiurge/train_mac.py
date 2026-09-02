# -*- coding: utf-8 -*-
"""
YOLOv8s 训练启动脚本（Mac Mini M2 Pro 版 - imgsz=1280 高质量版）
用法（在 Demiurge 目录下，使用 venv 的 python 运行）:
    source venv/bin/activate
    python train_mac.py

硬件：Mac Mini M2 Pro 10核 + 16核GPU + 32GB统一内存
device='mps' (Apple Metal Performance Shaders)

配置说明（不考虑训练时长，追求最佳质量）：
  ▸ imgsz=1280：高分辨率，精子小目标检测更准
  ▸ batch=8：保守值，避免 MPS 显存不足（32GB统一内存，但MPS管理有限）
  ▸ deterministic=False：关闭确定性模式，使 scatter_reduce/index_put 等算子可使用 MPS 加速（否则回退CPU导致极慢）
  ▸ epochs=200：充分训练，不赶时间
  ▸ patience=40：早停耐心值，防止过拟合
  ▸ cache=True：全数据集缓存到内存（32GB充裕），消除IO瓶颈
  ▸ close_mosaic=30：最后30轮关闭mosaic，精调阶段
  ▸ amp=False：MPS混合精度支持有限，关闭保稳定
  ▸ 其余超参保留已验证最优配置
"""
from ultralytics import YOLO

def main():
    # 1) 加载预训练模型 yolov8s
    model = YOLO("yolov8s.pt")

    # 2) Mac MPS 高质量版训练配置
    results = model.train(
        data="data_mac.yaml",
        # === 基础 ===
        model="yolov8s.pt",
        device='mps',                     # Apple M2 Pro GPU 加速
        project="/Users/xinmeiti/Desktop/entelecheia/Demiurge/runs/train",
        name="PoleMos600",                # 实验名：PoleMos600
        seed=0,
        val=True,
        amp=False,                         # MPS 混合精度支持有限，关闭保稳定
        plots=True,
        pretrained=True,
        cache=True,                        # 数据集缓存到内存（32GB充裕）
        # === 高质量版核心 ===
        imgsz=1280,                       # 高分辨率，小目标更准
        epochs=200,                       # 充分训练，不赶时间
        patience=40,                       # 早停，防止过拟合
        batch=8,                           # 保守值，避免MPS显存不足
        deterministic=False,                # 关闭确定性，使MPS算子可正常加速
        close_mosaic=30,                   # 最后30轮关闭mosaic精调
        # === 数据增强（保留最优）===
        copy_paste=0.15,
        mixup=0.2,
        mosaic=1.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        # === 优化器与损失（保留最优）===
        lr0=0.006,
        cos_lr=True,
        lrf=0.01,
        weight_decay=0.0005,
        box=8.0,
        cls=0.5,
        dfl=1.5,
        # === 其他 ===
        workers=8,
    )
    print("训练完成，权重与日志位于: runs/train/PoleMos600/")

if __name__ == "__main__":
    main()
