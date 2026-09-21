import numpy as np, cv2

ATT = r'c:\Users\86137\.trae-cn\attachments\6aa10629e61f710ea364bcf2\73390b9c-5471-4a7e-9787-f5ddea6722ae_3ca2014d-d3a0-4ec4-af5b-692789e83825_0001.bmp'
img = cv2.imdecode(np.fromfile(ATT, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
pol = bw.mean() < 127
bw2 = bw if pol else 255 - bw
n, lab, stats, cent = cv2.connectedComponentsWithStats(bw2)
ar = stats[1:, cv2.CC_STAT_AREA].astype(float)
xs, ys, ws, hs = stats[1:, 0], stats[1:, 1], stats[1:, 2], stats[1:, 3]
edge = (xs <= 1) | (ys <= 1) | (xs + ws >= img.shape[1] - 2) | (ys + hs >= img.shape[0] - 2)
m = (ar >= 100) & (ar <= 3000) & (~edge)
idx = np.where(m)[0] + 1
print(f'[精子头候选] 面积窗100-3000px且非贴边: {len(idx)} 个 (总面积范围 {ar[m].min():.0f}-{ar[m].max():.0f})')
L, S, EQ = [], [], []
for i in idx:
    p = (lab == i).astype(np.uint8)
    cnts, _ = cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: continue
    r = cnts[0].reshape(-1, 2).astype(float)
    r = r - r.mean(0)
    cov = r.T @ r / len(r)
    ev = np.sqrt(np.linalg.eigvalsh(cov)[::-1]) * np.sqrt(12)
    L.append(ev[0]); S.append(ev[1]); EQ.append(np.sqrt(ar[i - 1] * 4 / np.pi))
L, S, EQ = np.array(L), np.array(S), np.array(EQ)
k = (L > np.percentile(L, 5)) & (L < np.percentile(L, 95))
L, S, EQ = L[k], S[k], EQ[k]
u = 1000 / 3571.6
print(f'[几何统计px] 头长中位={np.median(L):.1f} 头宽中位={np.median(S):.1f} 等效径中位={np.median(EQ):.1f} (n={len(L)})')
print(f'[换算 µm/px={u:.4f}] 头长={np.median(L)*u:.2f}µm 头宽={np.median(S)*u:.2f}µm 等效径={np.median(EQ)*u:.2f}µm')
print('[参照] 猪头长7~10µm 宽4~6µm | 牛头长8~9.5µm | 人头长4~6.5µm')
