import cv2
import numpy as np

P_GRID = r'c:\Users\86137\.trae-cn\attachments\6aa10629e61f710ea364bcf2\e6ceb296-26d9-43b7-bf13-502e52b7d7f7_9bd98748-d502-4daa-8a6b-c475b9b9dd9e_0.bmp'
P_SPERM = r'c:\Users\86137\.trae-cn\attachments\6aa10629e61f710ea364bcf2\73390b9c-5471-4a7e-9787-f5ddea6722ae_3ca2014d-d3a0-4ec4-af5b-692789e83825_0001.bmp'

print('===== 方法1: 2D频谱直接找峰(Hann窗抗暗角, 峰位不受旋转影响) =====')
g = cv2.imread(P_GRID, cv2.IMREAD_GRAYSCALE).astype(np.float64)
H, W = g.shape
z = g - g.mean()
wy = np.hanning(H)[:, None]
wx = np.hanning(W)[None, :]
Z = z * wy * wx
F = np.fft.fft2(Z)
S = np.abs(np.fft.fftshift(F)) ** 2
cy, cx = H // 2, W // 2
S[cy - 8:cy + 9, cx - 8:cx + 9] = 0.0

def refine(ky0, kx0):
    def para(m1, m0, m2):
        d = m1 - 2 * m0 + m2
        return 0.5 * (m1 - m2) / d if abs(d) > 1e-12 else 0.0
    ls = np.log(S + 1e-12)
    dy = para(ls[ky0 - 1, kx0], ls[ky0, kx0], ls[ky0 + 1, kx0])
    dx = para(ls[ky0, kx0 - 1], ls[ky0, kx0], ls[ky0, kx0 + 1])
    return ky0 + dy, kx0 + dx

peaks = []
St = S.copy()
for it in range(2):
    ky0, kx0 = np.unravel_index(np.argmax(St), St.shape)
    ky, kx = refine(ky0, kx0)
    fy = (ky - cy) / H
    fx = (kx - cx) / W
    freq = np.hypot(fx, fy)
    period = 1.0 / freq
    angle = np.degrees(np.arctan2(fy, fx))
    peaks.append((period, angle, float(S[ky0, kx0])))
    print(f'峰{it + 1}: 周期={period:.2f}px 频率方向={angle:.2f}° (ky0={ky0},kx0={kx0})')
    St[ky0 - 6:ky0 + 7, kx0 - 6:kx0 + 7] = 0.0
    my, mx = int(round(2 * cy - ky0)), int(round(2 * cx - kx0))
    St[my - 6:my + 7, mx - 6:mx + 7] = 0.0
(p1, a1, s1), (p2, a2, s2) = peaks
perp = abs(abs(a1 - a2) - 90)
print(f'两峰周期差={abs(p1 - p2) / max(p1, p2) * 100:.2f}% 近垂直偏差={perp:.1f}°')
pitch_fft = (p1 + p2) / 2
harm_k = 2 * np.hypot((peaks[0][0] and 0), 0)

print('===== 方法2: 组件最近邻距离直方图取众数(抗对角外点) =====')
_, bw = cv2.threshold(g.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
n, lab, stats, cent = cv2.connectedComponentsWithStats(bw)
sel = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 5000]
c = cent[np.array(sel) - 1]
d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
np.fill_diagonal(d, np.inf)
nn = d.min(axis=1)
hist, edges = np.histogram(nn, bins=np.arange(150, 601, 5))
order = np.argsort(hist)[::-1][:3]
for r in order:
    print(f'直方图峰: {edges[r]:.0f}-{edges[r + 1]:.0f}px 计数={hist[r]}')
pitch_nn = float((edges[order[0]] + edges[order[0] + 1]) / 2)
inmod = nn[(nn >= edges[order[0]] - 7.5) & (nn <= edges[order[0] + 1] + 7.5)]
print(f'众数格距={pitch_nn:.1f}px 主峰±1bin样本数={len(inmod)} 样本内标准差={np.std(inmod):.2f}px')

print('===== 方法3: 自相关定向复核(在FFT峰±20%窗内, 盲搜已证不可靠) =====')
acs = []
for axis in (0, 1):
    prof = g.mean(axis=axis)
    p = prof - prof.mean()
    k = np.arange(len(p))
    cf = np.polyfit(k, p, 2)
    p = p - np.polyval(cf, k)
    f = np.fft.rfft(p)
    ac = np.fft.irfft(f * np.conj(f))
    ac /= ac[0]
    exp = pitch_fft
    lo, hi = int(exp * 0.8), int(exp * 1.2)
    lag0 = int(np.argmax(ac[lo:hi])) + lo
    y1, y2, y3 = ac[lag0 - 1], ac[lag0], ac[lag0 + 1]
    den = y1 - 2 * y2 + y3
    sub = 0.5 * (y1 - y3) / den if abs(den) > 1e-12 else 0.0
    acs.append(lag0 + sub)
    print(f'轴{axis}: 峰lag={lag0 + sub:.2f}px 峰值={y2:.3f} (搜索窗[{lo},{hi}])')
pitch_ac = float(np.mean(acs))

print('===== 三法对账 =====')
vals = [pitch_fft, pitch_nn, pitch_ac]
spread = (max(vals) - min(vals)) / np.median(vals) * 100
print(f'FFT={pitch_fft:.2f}px 最近邻众数={pitch_nn:.1f}px 自相关={pitch_ac:.2f}px 极差={spread:.2f}%')
if spread > 5:
    print('!! 三法分歧>5%: 不下结论')
    raise SystemExit(1)
pitch = pitch_fft
um_px = 100.0 / pitch
print(f'采用FFT均值: 格距={pitch:.2f}px / 0.1mm')
print(f'==> 标定: {um_px:.4f} um/px = {pitch * 10:.0f} px/mm')
print(f'==> 视野: {W * um_px / 1000:.3f}mm x {H * um_px / 1000:.3f}mm')
print('==> 光学放大倍率 M = 传感器像元 / um_px:')
for sname, s in (('IMX415(1.45um)', 1.45), ('IMX678(2.0um)', 2.0), ('IMX462/290(2.9um)', 2.9), ('IMX178(2.4um)', 2.4)):
    print(f'    {sname} -> M={s / um_px:.2f}x')

print('===== 精子头部旁证(按最终标定) =====')
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
