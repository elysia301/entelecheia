import cv2
import numpy as np

P_GRID = r'c:\Users\86137\.trae-cn\attachments\6aa10629e61f710ea364bcf2\e6ceb296-26d9-43b7-bf13-502e52b7d7f7_9bd98748-d502-4daa-8a6b-c475b9b9dd9e_0.bmp'
P_SPERM = r'c:\Users\86137\.trae-cn\attachments\6aa10629e61f710ea364bcf2\73390b9c-5471-4a7e-9787-f5ddea6722ae_3ca2014d-d3a0-4ec4-af5b-692789e83825_0001.bmp'

g = cv2.imread(P_GRID, cv2.IMREAD_GRAYSCALE)
H, W = g.shape

def para3(m1, m0, m2):
    d = m1 - 2 * m0 + m2
    return 0.5 * (m1 - m2) / d if abs(d) > 1e-12 else 0.0

print('===== 法1: 等长垂直峰对搜索(方格频谱指纹) =====')
z = g.astype(np.float64) - g.mean()
Z = z * np.hanning(H)[:, None] * np.hanning(W)[None, :]
F = np.fft.fftshift(np.fft.fft2(Z))
S = np.abs(F) ** 2
cy, cx = H // 2, W // 2
S[cy - 8:cy + 9, cx - 8:cx + 9] = 0.0
ky0, kx0 = np.unravel_index(np.argmax(S), S.shape)
ls = np.log(S + 1e-12)
ky = ky0 + para3(ls[ky0 - 1, kx0], ls[ky0, kx0], ls[ky0 + 1, kx0])
kx = kx0 + para3(ls[ky0, kx0 - 1], ls[ky0, kx0], ls[ky0, kx0 + 1])
fy1, fx1 = (ky - cy) / H, (kx - cx) / W
mag1 = float(np.hypot(fx1, fy1))
th1 = float(np.arctan2(fy1, fx1))
pitch_a = 1.0 / mag1
print(f'基频峰: 周期={pitch_a:.2f}px 方向={np.degrees(th1):.2f}° (峰值log={ls[ky0, kx0]:.2f})')

best = None
for s in (90, -90):
    th2e = th1 + np.radians(s)
    kye = cy + mag1 * np.sin(th2e) * H
    kxe = cx + mag1 * np.cos(th2e) * W
    ry, rx = int(0.06 * mag1 * H), int(0.06 * mag1 * W)
    y1, y2 = max(0, int(kye - ry)), min(S.shape[0], int(kye + ry) + 1)
    x1, x2 = max(0, int(kxe - rx)), min(S.shape[1], int(kxe + rx) + 1)
    sub = S[y1:y2, x1:x2].copy()
    sub[sub <= 0] = 0
    dj, di = np.unravel_index(np.argmax(sub), sub.shape)
    aj, ai = y1 + dj, x1 + di
    lsj = ls[aj - 1, ai], ls[aj, ai], ls[aj + 1, ai] if aj + 1 < H else (0, 0, 0)
    dy = para3(ls[aj - 1, ai], ls[aj, ai], ls[aj + 1, ai])
    dx = para3(ls[aj, ai - 1], ls[aj, ai], ls[aj, ai + 1])
    fy2, fx2 = (aj + dy - cy) / H, (ai + dx - cx) / W
    mag2 = float(np.hypot(fx2, fy2))
    th2 = float(np.arctan2(fy2, fx2))
    dev = abs(abs(np.degrees(th2 - th1)) - 90)
    if best is None or S[aj, ai] > best[0]:
        best = (S[aj, ai], mag2, th2, dev)
_, mag2, th2, dev = best
pitch_b = 1.0 / mag2
print(f'垂直搭档峰: 周期={pitch_b:.2f}px 方向={np.degrees(th2):.2f}° 垂直偏差={dev:.2f}°')
pair_diff = abs(pitch_a - pitch_b) / pitch_a * 100
print(f'峰对周期差={pair_diff:.2f}%')
pitch_fft = (pitch_a + pitch_b) / 2

print('===== 法2: 按基频方向转正后行/列剖面周期 =====')
results_rot = {}
for sign, label in ((1, '正转'), (-1, '反转')):
    M = cv2.getRotationMatrix2D((W / 2, H / 2), sign * np.degrees(th1), 1.0)
    rot = cv2.warpAffine(g, M, (W, H))
    pers = []
    for axis in (0, 1):
        p = rot.sum(axis=axis).astype(np.float64)
        sm = np.convolve(p, np.ones(601) / 601, mode='same')
        p = p - sm
        n = len(p)
        freqs = np.fft.rfftfreq(n)
        klo, khi = int(np.ceil(n / 470)), int(n / 250)
        sp = np.abs(np.fft.rfft(p))
        kk = int(np.argmax(sp[klo:khi + 1])) + klo
        kc = kk + para3(np.log(sp[kk - 1]), np.log(sp[kk]), np.log(sp[kk + 1]))
        pers.append(1.0 / (kc / n))
    results_rot[label] = pers
    print(f'{label}({sign * np.degrees(th1):.1f}°): 列向周期={pers[0]:.2f}px 行向周期={pers[1]:.2f}px 差={abs(pers[0] - pers[1]) / np.mean(pers) * 100:.2f}%')
ok_label = min(results_rot, key=lambda k: abs(results_rot[k][0] - results_rot[k][1]))
p0, p1_ = results_rot[ok_label]
pitch_rot = (p0 + p1_) / 2
print(f'采用{ok_label}: 转正双轴均值={pitch_rot:.2f}px (双轴差小说明转正正确、两方向格距一致)')

print('===== 法3: 几何法(面积窗+剔除贴边+迭代修剪) =====')
_, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
n, lab, stats, cent = cv2.connectedComponentsWithStats(bw)
big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 5000]
med_a = np.median([stats[i, cv2.CC_STAT_AREA] for i in big])
idx = []
for i in big:
    if not (0.6 * med_a <= stats[i, cv2.CC_STAT_AREA] <= 1.4 * med_a):
        continue
    if stats[i, cv2.CC_STAT_LEFT] <= 0 or stats[i, cv2.CC_STAT_TOP] <= 0:
        continue
    if stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] >= W:
        continue
    if stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] >= H:
        continue
    idx.append(i)
c = cent[np.array(idx) - 1]
d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
np.fill_diagonal(d, np.inf)
nn = d.min(axis=1)
m = float(np.median(nn))
for it in range(3):
    sel = np.abs(nn - m) < 0.08 * m
    m = float(np.median(nn[sel]))
print(f'完整单格={len(c)}个 迭代修剪后格距={m:.2f}px 入选样本={int(sel.sum())} 样本std={np.std(nn[sel]):.2f}px')

print('===== 三法对账 =====')
vals = [pitch_fft, pitch_rot, m]
spread = (max(vals) - min(vals)) / np.median(vals) * 100
print(f'FFT垂直峰对={pitch_fft:.2f}px 转正剖面={pitch_rot:.2f}px 几何修剪={m:.2f}px 极差={spread:.2f}%')
if spread > 4:
    print('!! 分歧>4% 不下结论')
    raise SystemExit(1)
pitch = float(np.mean(vals))
um_px = 100.0 / pitch
print(f'三法收敛: 格距={pitch:.2f}px / 0.1mm(标定卡Grid DIV标注)')
print(f'==> 标定: {um_px:.4f} um/px = {pitch * 10:.0f} px/mm = {um_px / 1000 * 1e0:.6f} mm/px')
print(f'==> 软件px_per_mm应设为 {pitch * 10:.0f}')
print(f'==> 视野: {W * um_px / 1000:.3f}mm x {H * um_px / 1000:.3f}mm (对角{np.hypot(W, H) * um_px / 1000:.3f}mm)')
print('==> 光学放大倍率 M = 像元尺寸/um_px:')
for sname, s in (('IMX415(1.45um,4K显微镜相机最常见)', 1.45), ('IMX678(2.0um)', 2.0), ('IMX178(2.4um)', 2.4), ('IMX462/290(2.9um)', 2.9), ('IMX174(5.86um)', 5.86)):
    print(f'    {sname} -> M={s / um_px:.2f}x')

print('===== 精子头部尺寸旁证(最终标定) =====')
g0 = cv2.imread(P_SPERM, cv2.IMREAD_GRAYSCALE)
bw0 = cv2.adaptiveThreshold(g0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 61, 10)
bw0 = cv2.morphologyEx(bw0, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
n0, lab0, st0, ce0 = cv2.connectedComponentsWithStats(bw0)
res = []
for i in range(1, n0):
    x, y, w, h, area = st0[i]
    if area < 300 or area > 3000:
        continue
    if x <= 0 or y <= 0 or x + w >= g0.shape[1] or y + h >= g0.shape[0]:
        continue
    comp = (lab0[y:y + h, x:x + w] == i).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        continue
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < area * 0.7:
        continue
    rw, rh = cv2.minAreaRect(cnt)[1]
    if min(rw, rh) < 5 or max(rw, rh) / min(rw, rh) > 2.0:
        continue
    res.append((max(rw, rh), min(rw, rh)))
res = np.array(res)
if len(res):
    L = res[:, 0] * um_px
    Sd = res[:, 1] * um_px
    print(f'紧凑头部候选={len(res)}个')
    print(f'长轴 中位={np.median(L):.2f}um P25-P75=[{np.percentile(L,25):.2f},{np.percentile(L,75):.2f}]')
    print(f'短轴 中位={np.median(Sd):.2f}um P25-P75=[{np.percentile(Sd,25):.2f},{np.percentile(Sd,75):.2f}]')
    print('(猪/牛精子头部长轴典型7-10um, 短轴4-6um; 落入区间=标定自洽)')
