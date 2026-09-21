import numpy as np, cv2

ATT = r'c:\Users\86137\.trae-cn\attachments\6aa10629e61f710ea364bcf2\e6ceb296-26d9-43b7-bf13-502e52b7d7f7_9bd98748-d502-4daa-8a6b-c475b9b9dd9e_0.bmp'
img = cv2.imdecode(np.fromfile(ATT, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
H, W = img.shape
bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
n, lab, stats, cent = cv2.connectedComponentsWithStats(bw)
xs, ys, ws, hs, ar = stats[1:, 0], stats[1:, 1], stats[1:, 2], stats[1:, 3], stats[1:, cv2.CC_STAT_AREA].astype(float)
edge = (xs <= 1) | (ys <= 1) | (xs + ws >= W - 2) | (ys + hs >= H - 2)
blk = (ar >= 30000) & (ar <= 500000) & (ws >= 100) & (hs >= 100) & (~edge)
P = cent[1:][blk]
print(f'[步骤1] 白方块本体 {int(blk.sum())} 个 (杂质点已被 宽高>=100 双重过滤排除; 面积范围 {ar[blk].min():.0f}-{ar[blk].max():.0f})')

D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
np.fill_diagonal(D, 1e9)
nn = D.min(1)
D0 = float(np.median(nn))
print(f'[步骤2] 最近邻中位 D0={D0:.2f}px (IQR {np.percentile(nn,25):.1f}-{np.percentile(nn,75):.1f})')

diff = P[:, None, :] - P[None, :, :]
dist = np.sqrt((diff ** 2).sum(-1))
ang = (np.degrees(np.arctan2(diff[..., 1], diff[..., 0])) % 180.0)
mask = (dist > 0.5 * D0) & (dist < 1.5 * D0)
ii, jj = np.where(np.triu(mask, 1))
a = ang[ii, jj]
hist, e = np.histogram(a, bins=36, range=(0, 180))
top2 = np.argsort(hist)[::-1][:2]
print('[步骤3] 排布方向角度直方图(5°bin) 前2峰:')
for b in top2:
    print(f'   {e[b]:5.1f}-{e[b+1]:5.1f}° : {"#"*int(hist[b])} {hist[b]}')

def dir_vec(theta_lo, theta_hi):
    m = mask & (ang >= theta_lo) & (ang < theta_hi)
    ii_, jj_ = np.where(np.triu(m, 1))
    L = dist[ii_, jj_]
    keep = L < np.percentile(L, 80)
    return L[keep]

def refine(lo, hi):
    L = dir_vec(lo, hi)
    med = np.median(L)
    for _ in range(3):
        L2 = L[np.abs(L - med) < 0.06 * med]
        if len(L2) >= 3: med = np.median(L2)
    return med, len(L)

b1 = e[top2[0]] + 2.5; b2 = e[top2[1]] + 2.5
D1, n1 = refine(b1 - 12, b1 + 12)
D2, n2 = refine(b2 - 12, b2 + 12)
print(f'[步骤4] 方向1 {b1:.1f}°: 格距={D1:.2f}px (n={n1}) | 方向2 {b2:.1f}°: 格距={D2:.2f}px (n={n2}) | 两方向差={abs(D1-D2)/min(D1,D2)*100:.2f}%')

cx, cy = W / 2, H / 2
r = np.sqrt((P[:, 0] - cx) ** 2 + (P[:, 1] - cy) ** 2)
rmax = np.sqrt(cx ** 2 + cy ** 2)
th = np.deg2rad(np.stack([b1, b2]))
print('[步骤5] 畸变检验(按半径3桶: 内圈/中圈/外圈):')
rads = []
for bi, (tlo, D_) in enumerate([(b1, D1), (b2, D2)]):
    rad = np.deg2rad(np.arange(tlo - 15, tlo + 16, 1.0))
    ok = ((rad[:, None] - np.deg2rad(ang)) % np.pi < np.deg2rad(1)).any(0) if False else None
    m = mask & (((ang - tlo + 15) % 180) <= 30)
    vals = {0: [], 1: [], 2: []}
    for p in range(len(P)):
        dd = dist[p][m[p]]
        if not len(dd): continue
        sel = dd[(dd > 0.7 * D_) & (dd < 1.3 * D_)]
        if len(sel):
            bucket = 0 if r[p] < rmax / 3 else (1 if r[p] < 2 * rmax / 3 else 2)
            vals[bucket].extend(sel.tolist())
    row = []
    for k in range(3):
        v = np.array(vals[k])
        row.append(f'桶{k}: {np.median(v):7.2f}px(n={len(v)})' if len(v) else f'桶{k}:   --')
    rads.append([np.median(np.array(vals[k])) if vals[k] else np.nan for k in range(3)])
    print(f'   方向{bi+1}({tlo:.0f}°): ' + ' | '.join(row))

d_c = np.nanmean([rads[0][0], rads[1][0]])
d_e = np.nanmean([rads[0][2], rads[1][2]])
print(f'[判定] 内圈均值={d_c:.2f}px 外圈均值={d_e:.2f}px 全场漂移={abs(d_e-d_c)/d_c*100:.2f}%')

overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for p in P:
    cv2.drawMarker(overlay, (int(p[0]), int(p[1])), (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
cv2.imwrite('_calib_tmp/diag_centroids.png', cv2.resize(overlay, (W // 2, H // 2)))
print('[输出] 质心十字验证图 -> _calib_tmp/diag_centroids.png')

for name, D_ in [('两方向联合中位', float(np.median(np.concatenate([dir_vec(b1-12, b1+12), dir_vec(b2-12, b2+12)])))),
                 ('内圈(中央)均值', d_c), ('全场均值', (D1 + D2) / 2)]:
    pxmm = D_ * 10
    umpx = 1000 / pxmm
    print(f'\n=== {name}: 格距={D_:.2f}px ===')
    print(f'    px/mm={pxmm:.1f}  µm/px={umpx:.4f}  视野={W/pxmm:.3f}x{H/pxmm:.3f}mm')
    for sn, px in [('IMX415(1.45)', 1.45), ('IMX678(2.0)', 2.0), ('IMX178(2.4)', 2.4),
                   ('IMX462(2.9)', 2.9), ('IMX290(2.9)', 2.9), ('IMX174(5.86)', 5.86)]:
        print(f'    {sn}µm -> M={px/umpx:6.1f}x')
