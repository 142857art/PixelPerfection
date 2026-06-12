import cv2
import numpy as np
import os
from perfect_pixel.perfect_pixel import detect_grid_scale, refine_grids
from sample_palette_mode import sample_palette_mode

IMG_PATH = "images/avatar.png"

# --- 한글 경로 안전 읽기 ---
if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(
        f"파일이 없습니다: {os.path.abspath(IMG_PATH)}\n"
        "run.py가 있는 폴더에 images 폴더를 만들고 그 안에 avatar.png를 넣으세요."
    )
data = np.fromfile(IMG_PATH, dtype=np.uint8)   # 한글 경로 OK
bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
if bgr is None:
    raise RuntimeError("파일은 있지만 이미지 디코딩에 실패했습니다. PNG 파일이 맞는지 확인하세요.")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# --- 이하 동일 ---
grid_w, grid_h = detect_grid_scale(rgb, peak_width=6, max_ratio=1.5, min_size=4.0)
if grid_w is None:
    raise RuntimeError("그리드 검출 실패 — grid_w, grid_h를 수동으로 지정하세요.")

x_coords, y_coords = refine_grids(rgb, grid_w, grid_h, refine_intensity=0.25)
out = sample_palette_mode(rgb, x_coords, y_coords, n_colors=16, margin=0.25)
print(f"출력 크기: {out.shape[1]} x {out.shape[0]}")

# --- 한글 경로 안전 저장 ---
def imwrite_kr(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if ok:
        buf.tofile(path)

imwrite_kr("output_tiny.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
big = cv2.resize(out, None, fx=16, fy=16, interpolation=cv2.INTER_NEAREST)
imwrite_kr("output_big.png", cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
print("저장 완료: output_tiny.png, output_big.png")