import cv2
import numpy as np

P_GRID = r'c:\Users\86137\.trae-cn\attachments\6aa10629e61f710ea364bcf2\e6ceb296-26d9-43b7-bf13-502e52b7d7f7_9bd98748-d502-4daa-8a6b-c475b9b9dd9e_0.bmp'
P_SPERM = r'c:\Users\86137\.trae-cn\attachments\6aa10629e61f710ea364bcf2\73390b9c-5471-4a7e-9787-f5ddea6722ae_3ca2014d-d3a0-4ec4-af5b-692789e83825_0001.bmp'

print('===== A. 完整单格过滤后重测格距 =====')
g = cv2.imread(P_GRID, cv2.IMREAD_GRAYSCALE)
H, W = g.shape
_, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
n, lab, stats, cent = cv2.connectedComponentsWithStats(bw)
areas = stats[1:, cv2.CC_STAT_AREA]
big = areas[areas > 5000]
med_a = np.median(big)
fill_lo, fill_hi = 0.6 * med_a, 1.3 * med_a
ratio = stats[1:, cv2.CC_STAT_WIDTH] / np.maximum(stats[1:, cv2.CC_STAT_HEIGHT], 1)
keep = []
for i in range(1, n):
    if not (5000 < areas[i - 1] and fill_lo <= areas[i - 1] <= fill_hi):
        continue
    if not (0.75 <= ratio[i - 1] <= 1.33):
        continue
    if stats[i, cv2.CC_STAT_LEFT] <= 0 or stats[i, cv2.CC_STAT_TOP] <= 0:
        continue
    if stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] >= W:
        continue
    if stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] >= H:
        continue
    keep.append(i)
keep = np.array(keep)
c = cent[keep - 1]
print(f'单格候选: {len(keep)}个 面积窗[{fill_lo:.0f},{fill_hi:.0f}] 中位面积={med_a:.0f}px 等效边长={np.sqrt(med_a):.1f}px')
d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
np.fill_diagonal(d, np.inf)
nn = d.min(axis=1)
pitch_nn = float(np.median(nn))
mad = float(np.median(np.abs(nn - pitch_nn)))
vec_mad = mad / pitch_nn * 100
print(f'过滤后格距中位数={pitch_nn:.2f}px MAD={mad:.2f}px 相对波动={vec_mad:.2f}%')
ang = (np.degrees(np.arctan2((c[np.nanargmin(d, axis=1)] - c)[:, 1], (c[np.nanargmin(d, axis=1)] - c)[:, 0]))) % 180
ang_m = float(np.median(ang))

print('===== B. 自相关法交叉验证（去趋势，旋转对齐后按行/列剖面）=====')
M = cv2.getRotationMatrix2D((W / 2, H / 2), -ang_m, 1.0)
rot = cv2.warpAffine(bw, M, (W, H))
ac = {}
for axis, name in ((0, 'x'), (1, 'y')):
    prof = rot.sum(axis=axis).astype(np.float64)
    p = prof - prof.mean()
    sm = np.convolve(p, np.ones(801) / 801, mode='same')
    p = p - sm
    p[:400] = 0
    p[-400:] = 0
    f = np.fft.rfft(p)
    acf = np.fft.irfft(f * np.conj(f))
    acf /= acf[0]
    band = acf[150:600]
    lag0 = int(np.argmax(band)) + 150
    y1, y2, y3 = acf[lag0 - 1], acf[lag0], acf[lag0 + 1]
    denom = (y1 - 2 * y2 + y3)
    sub = 0.5 * (y1 - y3) / denom if abs(denom) > 1e-12 else 0.0
    lag = lag0 + sub
    ac[name] = lag
    print(f'{name}向 自相关峰 lag={lag:.2f}px (整数峰{lag0}, 峰值={y2:.3f})')
pitch_fft2 = (ac['x'] + ac['y']) / 2

print('===== C. 两法合成最终标定 =====')
w_nn = 1.0 / max(vec_mad, 1e-6) ** 2
pitch = (pitch_nn * w_nn + pitch_fft2 * 1.0) / (w_nn + 1.0)
print(f'倾角={ang_m:.2f}° 最近邻={pitch_nn:.2f}px 自相关均值={pitch_fft2:.2f}px 合成={pitch:.2f}px / 0.1mm')
um_px = 100.0 / pitch
print(f'==> 标定值: {um_px:.4f} um/px  ({pitch * 10:.0f} px/mm)')
print(f'==> 视野: {W * um_px / 1000:.3f}mm x {H * um_px / 1000:.3f}mm')
sensor = [('IMX415 4K显微镜相机常用', 1.45), ('IMX678', 2.0), ('IMX462/290类', 2.9), ('IMX178', 2.4), ('IMX226', 1.86)]
print('==> 若按常见传感器像元折算光学放大倍率:')
for sname, s in sensor:
    print(f'    {sname} {s}um -> M = {s / um_px:.2f}x')

print('===== D. 精子头部旁证（复核）=====')
g0 = cv2.imread(P_SPERM, cv2.IMREAD_GRAYSCALE)
bw0 = cv2.adaptiveThreshold(g0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 61, 10)
bw0 = cv2.morphologyEx(bw0, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
n0, lab0, st0, ce0 = cv2.connectedComponentsWithStats(bw0)
res = []
for i in range(1, n0):
    x, y, w, h, area = st0[i]
    if area < 400 or area > 2000:
        continue
    if x <= 0 or y <= 0 or x + w >= g0.shape[1] or y + h >= g0.shape[0]:
        continue
    comp = (lab0[y:y + h, x:x + w] == i).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        continue
    cnt = max(cnts, key=cv2.contourArea)
    ca = cv2.contourArea(cnt)
    if ca < area * 0.75:
        continue
    rw, rh = cv2.minAreaRect(cnt)[1]
    if min(rw, rh) < 6:
        continue
    if max(rw, rh) / min(rw, rh) > 2.2:
        continue
    res.append((max(rw, rh), min(rw, rh)))
res = np.array(res)
if len(res):
    L = res[:, 0] * um_px
    Sd = res[:, 1] * um_px
    print(f'头部候选(近椭圆紧凑群)={len(res)}个')
    print(f'长轴 中位={np.median(L):.2f}um P10-P90=[{np.percentile(L,10):.2f},{np.percentile(L,90):.2f}]')
    print(f'短轴 中位={np.median(Sd):.2f}um P10-P90=[{np.percentile(Sd,10):.2f},{np.percentile(Sd,90):.2f}]')
