import numpy as np, cv2

ATT = r'c:\Users\86137\.trae-cn\attachments\6aa10629e61f710ea364bcf2\e6ceb296-26d9-43b7-bf13-502e52b7d7f7_9bd98748-d502-4daa-8a6b-c475b9b9dd9e_0.bmp'
img = cv2.imdecode(np.fromfile(ATT, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
H, W = img.shape
print(f'[步骤1] 载入 {W}x{H}')

bw0 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
pol = 1 if bw0.mean() < 127 else -1
bw = bw0 if pol == 1 else 255 - bw0
n, lab, stats, cent = cv2.connectedComponentsWithStats(bw)
areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
xs, ys, ws, hs = stats[1:, 0], stats[1:, 1], stats[1:, 2], stats[1:, 3]
edge = (xs <= 1) | (ys <= 1) | (xs + ws >= W - 2) | (ys + hs >= H - 2)
keep = (areas >= 80) & (areas <= 50000) & (~edge)
P = cent[1:][keep]
print(f'[步骤2] 非贴边+面积窗组件 {int(keep.sum())}/{n-1} 个')
D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
np.fill_diagonal(D, 1e9)
nn = D.min(1)
print('[证据A] 最近邻距离完整直方图(40px箱):')
hist, edges = np.histogram(nn, bins=np.arange(0, nn.max() + 40, 40))
for h, e in zip(hist, edges):
    if h:
        print(f'   {e:5.0f}-{e+40:5.0f}px : {"#"*int(h)} {h}')
print(f'   排序值: {np.array2string(np.sort(nn), precision=0, max_line_width=250)}')

def project(src, d, theta_deg):
    th = np.deg2rad(theta_deg); c, s = np.cos(th), np.sin(th)
    xs_, ys_ = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    t = xs_ * c + ys_ * s
    tmin = t.min()
    off = -int(np.floor(tmin))
    ti = np.round(t + off).astype(np.int64)
    acc = np.bincount(ti.ravel(), weights=src.astype(np.float64).ravel(), minlength=ti.max() + 2)
    cnt = np.bincount(ti.ravel(), minlength=ti.max() + 2)
    prof = acc / np.maximum(cnt, 1)
    best = (-1.0, 0.0)
    for ph in np.arange(0, d, 1.0):
        idx = (np.arange(off, ti.max() + 1, d) - off + ph).astype(int)
        v = prof[idx].sum()
        if v > best[0]: best = (v, ph)
    return best[1] - off

def draw_lines(canvas, d, theta_deg, ph0, color, lw=4):
    th = np.deg2rad(theta_deg); c, s = np.cos(th), np.sin(th)
    t_lo = min(0, W * c) + min(0, H * s) - d
    t_hi = max(0, W * c) + max(0, H * s) + d
    k = int(np.floor(t_lo / d))
    while k * d + ph0 <= t_hi:
        t0 = k * d + ph0
        p1 = (int(t0 * c - 6000 * s), int(t0 * s + 6000 * c))
        p2 = (int(t0 * c + 6000 * s), int(t0 * s - 6000 * c))
        cv2.line(canvas, p1, p2, color, lw)
        k += 1

src = img if pol == 1 else 255 - img
print(f'[步骤3] 极性={"亮线" if pol==1 else "暗线"}, 生成三假设叠加图')
col = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
ph1 = project(src, 358.99, -27.39); draw_lines(col, 358.99, -27.39, ph1, (0, 0, 255), 4)
ph2 = project(src, 393.95, 65.77);  draw_lines(col, 393.95, 65.77, ph2, (255, 0, 0), 4)
ph3 = project(src, 194.28, -27.39); draw_lines(col, 194.28, -27.39, ph3, (0, 255, 0), 2)
cv2.imwrite('_calib_tmp/diag_overlay.png', cv2.resize(col, (W // 2, H // 2)))
print('   红=FFT基频358.99px | 蓝=FFT搭档393.95px | 绿=几何194.28px -> _calib_tmp/diag_overlay.png')

def peak_count(prof):
    p = prof.astype(np.float64); p = p - p.mean(); sd = p.std()
    pk = []
    for i in range(2, len(p) - 2):
        if p[i] >= p[i-1] and p[i] >= p[i+1] and p[i] > sd * 1.5:
            if pk and i - pk[-1] < 60:
                if p[i] > p[pk[-1]]: pk[-1] = i
            else:
                pk.append(i)
    d = np.diff(pk)
    return len(pk), (float(np.median(d)) if len(d) else 0.0)

th = np.deg2rad(-27.39); c, s = np.cos(th), np.sin(th)
M = np.array([[c, s, 0], [-s, c, 0]], dtype=np.float64)
nw_, nh_ = int(W * abs(c) + H * abs(s)) + 2, int(W * abs(s) + H * abs(c)) + 2
M[0, 2] = nw_ / 2 - (c * W / 2 + s * H / 2)
M[1, 2] = nh_ / 2 - (-s * W / 2 + c * H / 2)
rot = cv2.warpAffine(img, M, (nw_, nh_), flags=cv2.INTER_LINEAR, borderValue=float(img.mean()))
print('[证据B] 转正(-27.39°)后数格线:')
for tag, rr in [('亮线峰', rot.astype(np.float64)), ('暗线峰', 255.0 - rot.astype(np.float64))]:
    nc, dc = peak_count(rr.mean(axis=0))
    nr, dr = peak_count(rr.mean(axis=1))
    print(f'   {tag}: 列向 {nc} 条 间距中位 {dc:.1f}px | 行向 {nr} 条 间距中位 {dr:.1f}px')
