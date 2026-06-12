# -*- coding: utf-8 -*-
"""
perfectPixel 보조 모듈 v2
  - build_palette_exact : 원본에 '실제 존재하는' 색만 빈도순으로 채택 (색감 시프트 없음, 색 수 자동)
  - sample_palette_mode : 셀 마진 크롭 + 중심 가중 다수결로 팔레트 인덱스 투표
  - detect_grid_harmonic: 서브하모닉(절반 해상도) 검출 자동 교정
"""
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ---------------- 팔레트: 실제 존재 색 기반 (색감 보존) ----------------
def build_palette_exact(samples, max_colors=0, min_color_dist=24.0, min_count=2):
    """
    samples: (N, C) 셀 중심에서 뽑은 색들.
    1) 8단위 빈으로 묶어 빈도 랭킹(노이즈로 흩어진 같은 색을 합산)
    2) 빈도 높은 빈부터, 기존 팔레트와 min_color_dist 이상 떨어져 있으면 채택
    3) 채택 색 = 그 빈 안에서 '가장 많이 등장한 실제 픽셀 색' (평균 아님!)
    max_colors=0 이면 개수 제한 없음(자동).
    """
    s = samples.reshape(-1, samples.shape[-1]).astype(np.float32)
    b = np.clip((s // 8).astype(np.int64), 0, 31)
    keys = b[:, 0] * 1024 + b[:, 1] * 32 + b[:, 2]
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)

    palette = []
    for oi in order:
        if counts[oi] < min_count:
            break
        members = s[inv == oi].astype(np.uint8)
        cols, ccnt = np.unique(members.reshape(-1, members.shape[-1]), axis=0, return_counts=True)
        rep = cols[ccnt.argmax()].astype(np.float32)   # 실제 존재하는 색
        if all(np.sqrt(((rep - p) ** 2).sum()) >= min_color_dist for p in palette):
            palette.append(rep)
            if max_colors and len(palette) >= max_colors:
                break
    if not palette:                                     # 안전장치
        palette = [s.mean(axis=0)]
    return np.stack(palette).astype(np.float32)


def quantize_to_palette(image, palette, chunk=200_000):
    H, W = image.shape[:2]
    px = image.reshape(-1, image.shape[-1]).astype(np.float32)
    idx = np.empty(px.shape[0], dtype=np.int32)
    for st in range(0, px.shape[0], chunk):
        block = px[st:st + chunk]
        d = ((block[:, None, :] - palette[None, :, :]) ** 2).sum(-1)
        idx[st:st + chunk] = d.argmin(1)
    return idx.reshape(H, W)


def _center_samples(image, x_coords, y_coords):
    H, W = image.shape[:2]
    cx = np.clip(((np.asarray(x_coords)[1:] + np.asarray(x_coords)[:-1]) * 0.5), 0, W - 1).astype(np.int32)
    cy = np.clip(((np.asarray(y_coords)[1:] + np.asarray(y_coords)[:-1]) * 0.5), 0, H - 1).astype(np.int32)
    return image[cy[:, None], cx[None, :]].reshape(-1, image.shape[-1])


def sample_palette_mode(image, x_coords, y_coords,
                        n_colors=0, palette=None,
                        min_color_dist=24.0,
                        margin=0.25, sigma_frac=0.4):
    """
    n_colors=0 → 팔레트 색 수 자동 (실존 색 기반).
    margin    : 셀 4변에서 잘라내는 비율(오염된 경계 제외). 0.25 → 안쪽 50%만 투표.
    sigma_frac: 중심 가중치의 퍼짐 정도(작을수록 중심 픽셀 위주).
    """
    H, W = image.shape[:2]
    if palette is None:
        cs = _center_samples(image, x_coords, y_coords)
        palette = build_palette_exact(cs, max_colors=int(n_colors), min_color_dist=min_color_dist)
    K = palette.shape[0]
    index_map = quantize_to_palette(image, palette)

    x = np.asarray(x_coords, dtype=np.float64)
    y = np.asarray(y_coords, dtype=np.float64)
    nx, ny = len(x) - 1, len(y) - 1
    out = np.empty((ny, nx, palette.shape[1]), dtype=np.float32)

    for j in range(ny):
        cy0, cy1 = y[j], y[j + 1]
        my = (cy1 - cy0) * margin
        y0 = int(np.clip(np.floor(cy0 + my), 0, H - 1))
        y1 = int(np.clip(np.ceil(cy1 - my), y0 + 1, H))
        for i in range(nx):
            cx0, cx1 = x[i], x[i + 1]
            mx = (cx1 - cx0) * margin
            x0 = int(np.clip(np.floor(cx0 + mx), 0, W - 1))
            x1 = int(np.clip(np.ceil(cx1 - mx), x0 + 1, W))

            cell = index_map[y0:y1, x0:x1]
            h, w = cell.shape
            yy = (np.arange(h) - (h - 1) / 2.0) / max(h * sigma_frac, 1e-6)
            xx = (np.arange(w) - (w - 1) / 2.0) / max(w * sigma_frac, 1e-6)
            wgt = np.exp(-0.5 * (yy[:, None] ** 2 + xx[None, :] ** 2))
            votes = np.bincount(cell.ravel(), weights=wgt.ravel(), minlength=K)
            out[j, i] = palette[int(votes.argmax())]

    if image.dtype == np.uint8:
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out


# ---------------- 서브하모닉(절반 해상도) 자동 교정 ----------------
def _recon_error(image, gw, gh, refine_fn, refine_intensity=0.25):
    """grid (gw,gh)로 center 샘플링 후 원본 크기로 복원했을 때의 평균 오차."""
    x, y = refine_fn(image, gw, gh, refine_intensity)
    cx = np.clip(((np.asarray(x)[1:] + np.asarray(x)[:-1]) * 0.5), 0, image.shape[1] - 1).astype(np.int32)
    cy = np.clip(((np.asarray(y)[1:] + np.asarray(y)[:-1]) * 0.5), 0, image.shape[0] - 1).astype(np.int32)
    tiny = image[cy[:, None], cx[None, :]]
    if _HAS_CV2:
        up = cv2.resize(tiny, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        ry = (np.arange(image.shape[0]) * tiny.shape[0] // image.shape[0]).clip(0, tiny.shape[0] - 1)
        rx = (np.arange(image.shape[1]) * tiny.shape[1] // image.shape[1]).clip(0, tiny.shape[1] - 1)
        up = tiny[ry[:, None], rx[None, :]]
    return float(np.abs(up.astype(np.float32) - image.astype(np.float32)).mean()), (x, y)


def detect_grid_harmonic(image, gw, gh, refine_fn,
                         min_cell=3.0, improve_ratio=0.80, refine_intensity=0.25):
    """
    검출된 (gw,gh)에서 시작해 그리드를 2배씩 올려보며 복원 오차가
    improve_ratio 미만으로 '크게' 줄어드는 동안 채택.
    (진짜 그리드가 g면 2g는 오차 개선이 미미, 진짜가 2g였다면 크게 개선됨)
    반환: (gw, gh, doubled_times)
    """
    H, W = image.shape[:2]
    err, _ = _recon_error(image, gw, gh, refine_fn, refine_intensity)
    doubled = 0
    while True:
        gw2, gh2 = gw * 2, gh * 2
        if W / gw2 < min_cell or H / gh2 < min_cell:
            break
        e2, _ = _recon_error(image, gw2, gh2, refine_fn, refine_intensity)
        if e2 < err * improve_ratio:
            gw, gh, err = gw2, gh2, e2
            doubled += 1
        else:
            break
    return gw, gh, doubled