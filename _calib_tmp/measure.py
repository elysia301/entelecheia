import cv2
import numpy as np

P_GRID = r'c:\Users\86137\.trae-cn\attachments\6aa10629e61f710ea364bcf2\e6ceb296-26d9-43b7-bf13-502e52b7d7f7_9bd98748-d502-4daa-8a6b-c475b9b9dd9e_0.bmp'
P_SPERM = r'c:\Users\86137\.trae-cn\attachments\6aa10629e61f710ea364bcf2\73390b9c-5471-4a7e-9787-f5ddea6722ae_3ca2014d-d3a0-4ec4-af5b-692789e83825_0001.bmp'

print('===== 步骤1: 网格图连通域提取 =====')
g = cv2.imread(P_GRID, cv2.IMREAD_GRAYSCALE)
H, W = g.shape
_, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
otsu_val = float(np.mean(g[bw > 0])), float(np.mean(g[bw == 0]))
print(f'Otsu二值化: 亮区均值={otsu_val[0]:.1f} 暗区均值={otsu_val[1]:.1f} 亮区占比={np.mean(bw>0)*100:.1f}%')
bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
n, lab, stats, cent = cv2.connectedComponentsWithStats(bw)
areas_all = stats[1:, cv2.CC_STAT_AREA]
sel = areas_all > 5000
c = cent[1:][sel]
a = areas_all[sel]
print(f'连通域总数={n-1} 方格级组件={len(c)} 组件面积中位数={np.median(a):.0f}px 等效边长={np.sqrt(np.median(a)):.1f}px')

print('===== 步骤2: 最近邻间距（格子中心距=格距）=====')
d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
np.fill_diagonal(d, np.inf)
nn = d.min(axis=1)
j = d.argmin(axis=1)
pitch = float(np.median(nn))
mad = float(np.median(np.abs(nn - pitch)))
vec = c[j] - c
ang = (np.degrees(np.arctan2(vec[:, 1], vec[:, 0]))) % 180
ang_m = float(np.median(ang))
lo, hi = np.percentile(nn, [10, 90])
print(f'格距(最近邻中位数)={pitch:.2f}px  MAD={mad:.2f}px  P10-P90=[{lo:.1f},{hi:.1f}]')
print(f'网格倾角中位数={ang_m:.2f}°')

print('===== 步骤3: 剖面FFT交叉验证 =====')
M = cv2.getRotationMatrix2D((W / 2, H / 2), -ang_m, 1.0)
rot = cv2.warpAffine(bw, M, (W, H))
res = {}
for axis, name in ((0, 'x'), (1, 'y')):
    prof = rot.sum(axis=axis).astype(float)
    prof = prof - prof.mean()
    sp = np.abs(np.fft.rfft(prof))
    k = int(np.argmax(sp[1:]) + 1)
    spacing = 1.0 / np.fft.rfftfreq(len(prof))[k]
    res[name] = spacing
    print(f'{name}向剖面主频周期={spacing:.2f}px')
fft_mean = (res['x'] + res['y']) / 2

print('===== 步骤4: 换算（假设格距=0.1mm，与标定卡Grid Shaped DIV=0.1mm标注一致）=====')
for tag, ppx in (('最近邻法', pitch), ('FFT均值', fft_mean)):
    um_px = 100.0 / ppx
    px_mm = ppx * 10.0
    print(f'[{tag}] 格距={ppx:.2f}px/0.1mm -> {px_mm:.0f} px/mm -> {um_px:.4f} um/px')
    print(f'       视野={W * um_px / 1000:.2f}mm x {H * um_px / 1000:.2f}mm')
    for sname, s in (('IMX415(1.45um)', 1.45), ('1.8um类', 1.8), ('2.0um类', 2.0), ('3.0um类', 3.0)):
        print(f'       若传感器像元{s}: 光学放大倍率 M={s / um_px:.1f}x')

print('===== 步骤5: 精子头部尺寸旁证（bmp0）=====')
g0 = cv2.imread(P_SPERM, cv2.IMREAD_GRAYSCALE)
bw0 = cv2.adaptiveThreshold(g0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 61, 10)
bw0 = cv2.morphologyEx(bw0, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
n0, lab0, st0, ce0 = cv2.connectedComponentsWithStats(bw0)
maj = []
for i in range(1, n0):
    x, y, w, h, area = st0[i]
    if area < 150 or area > 8000:
        continue
    if x <= 0 or y <= 0 or x + w >= g0.shape[1] or y + h >= g0.shape[0]:
        continue
    comp = (lab0[y:y + h, x:x + w] == i).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        continue
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < area * 0.55:
        continue
    rw, rh = cv2.minAreaRect(cnt)[1]
    if min(rw, rh) < 3:
        continue
    maj.append((max(rw, rh), min(rw, rh)))
maj = np.array(maj)
print(f'候选头部数={len(maj)}')
if len(maj):
    um_px = 100.0 / pitch
    L = maj[:, 0] * um_px
    Sd = maj[:, 1] * um_px
    print(f'长轴: 中位数={np.median(L):.1f}um  P10-P90=[{np.percentile(L,10):.1f},{np.percentile(L,90):.1f}]')
    print(f'短轴: 中位数={np.median(Sd):.1f}um  P10-P90=[{np.percentile(Sd,10):.1f},{np.percentile(Sd,90):.1f}]')
