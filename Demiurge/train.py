# -*- coding: utf-8 -*-
"""
YOLOv8s 训练启动脚本（640快速试水版 v5：imgsz=640 + batch=16，目标<2小时）
用法（在 Demiurge 目录下，使用 venv 的 python 运行）:
    venv\\Scripts\\python.exe train.py
【注意】本文件仅修改参数，未启动训练。按需求手动执行上面的命令。

硬件：RTX 3050 Laptop 4GB → model=yolov8s.pt, device=0, batch=16, amp=True, imgsz=640

640快速版改动依据（在快速版v4基础上，进一步压缩到<2小时）：
  【改动1】imgsz 960→640：计算量降到44%，加速2.25倍
      质量影响：精子目标从15px→10px，接近检测下限，mAP50预计降15~25%
      用途：快速试水、调参验证、原型测试，不做最终模型
  【改动2】batch 8→16：每轮迭代数119→60，减50%
      显存验证：640×640×16=6.55M像素 = 原1280×1280×4，4GB显存可运行
  【改动3】epochs 120→100：单轮更快，100轮约1.5~1.8小时
  【改动4】patience 30→20：更灵敏的早停
  【改动5】close_mosaic 15→10：最后10轮关闭mosaic

保留不动（已验证最优）：
  ▸ lr0=0.006 / cos_lr / lrf=0.01 / cls=0.5 / dfl=1.5 / box=8.0
  ▸ copy_paste=0.15 / mixup=0.2 / mosaic=1.0 / fliplr=0.5
  ▸ hsv_h=0.015 / hsv_s=0.5 / hsv_v=0.4
  ▸ translate=0.1 / scale=0.5 / degrees=shear=perspective=flipud=0
  ▸ weight_decay=0.0005 / workers=8 / seed=0 / pretrained=True / amp=True / cache=True
  ▸ data.yaml / project 路径

预计耗时：单轮约1.0~1.3分钟，100轮约1.7~2.2小时（早停触发则更短）
"""
from ultralytics import YOLO

def main():
    # 1) 加载预训练模型 yolov8s
    model = YOLO("yolov8s.pt")

    # 2) 640快速试水版训练配置
    results = model.train(
        data="data.yaml",
        # === 基础 ===
        model="yolov8s.pt",
        device=0,
        project=r"C:/Users/86137/Desktop/entelecheia/Demiurge/runs/train",
        name="KaLos618_640",            # 640快速试水版，不覆盖之前的实验
        seed=0,
        val=True,
        amp=True,
        plots=True,
        pretrained=True,
        cache=True,                      # 缓存数据集，消除IO瓶颈
        # === 640快速版核心 ===
        imgsz=640,                      # 改动1: 960→640，计算量降44%
        epochs=100,                     # 改动3: 120→100，目标<2小时
        patience=20,                     # 改动4: 30→20，早停更灵敏
        batch=16,                        # 改动2: 8→16，迭代数减半
        close_mosaic=10,                 # 改动5: 15→10，尾段关闭mosaic
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
    print("训练完成，权重与日志位于: runs/train/KaLos618_640/")

if __name__ == "__main__":
    main()
