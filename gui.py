import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import sys
import re
from datetime import datetime

# Conditional import for DnD to allow it to fail gracefully if not installed (though we installed it)
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False

# Ensure src is in python path to import perfect_pixel
# PyInstaller unpacks data to sys._MEIPASS
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

src_dir = os.path.join(base_path, 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

try:
    from perfect_pixel.perfect_pixel import (
        detect_grid_scale, refine_grids,
        sample_center, sample_median, sample_majority,
    )
except ImportError:
    from src.perfect_pixel.perfect_pixel import (
        detect_grid_scale, refine_grids,
        sample_center, sample_median, sample_majority,
    )

from sample_palette_mode import sample_palette_mode, detect_grid_harmonic


# ---------- 한글 경로 안전 I/O ----------
def imread_kr(path):
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def imwrite_kr(path, rgb):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png",
                           cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if ok:
        buf.tofile(path)
        return True
    return False


# ---------- 처리 파이프라인 (get_perfect_pixel 대체) ----------
def run_pipeline(rgb, method="palette-mode",
                 grid_w=0, grid_h=0, harmonic=True,
                 refine_intensity=0.25,
                 n_colors=0, min_color_dist=24.0, margin=0.25,
                 fix_square=True):
    """returns (w, h, out_image, info_str) — 실패 시 (None, None, None, msg)"""
    notes = []
    if grid_w and grid_h:
        gw, gh = int(grid_w), int(grid_h)
        notes.append(f"grid manual {gw}x{gh}")
    else:
        gw, gh = detect_grid_scale(rgb, peak_width=6, max_ratio=1.5, min_size=4.0)
        if gw is None or gh is None:
            return None, None, None, "Grid detection failed. Set manual grid."
        notes.append(f"grid auto {gw}x{gh}")
        if harmonic:
            gw2, gh2, doubled = detect_grid_harmonic(rgb, gw, gh, refine_grids,
                                                     refine_intensity=refine_intensity)
            if doubled:
                notes.append(f"harmonic x{2**doubled} -> {gw2}x{gh2}")
                gw, gh = gw2, gh2

    x_coords, y_coords = refine_grids(rgb, gw, gh, refine_intensity)

    if method == "palette-mode":
        out = sample_palette_mode(rgb, x_coords, y_coords,
                                  n_colors=int(n_colors),
                                  min_color_dist=float(min_color_dist),
                                  margin=float(margin))
    elif method == "median":
        out = sample_median(rgb, x_coords, y_coords)
    elif method == "majority":
        out = sample_majority(rgb, x_coords, y_coords)
    else:
        out = sample_center(rgb, x_coords, y_coords)

    rx, ry = out.shape[1], out.shape[0]
    # fix_square (레포 로직 이식)
    if fix_square and abs(rx - ry) == 1:
        if rx > ry:
            if rx % 2 == 1:
                out = out[:, :-1]
            else:
                out = np.concatenate([out[:1, :], out], axis=0)
        else:
            if ry % 2 == 1:
                out = out[:-1, :]
            else:
                out = np.concatenate([out[:, :1], out], axis=1)
    h, w = out.shape[:2]
    return w, h, out, " | ".join(notes)

class PerfectPixelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Perfect Pixel GUI")
        self.root.geometry("1100x700")

        # Variables
        self.current_image_path = None
        self.original_cv_image = None
        self.processed_cv_image = None
        self.display_image = None 
        self.viewing = "original"
        
        self.canvas_image_id = None
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start_x = 0
        self.drag_start_y = 0

        # UI Layout
        self._create_layout()

        # Drag and Drop Registration
        if HAS_DND and hasattr(self.root, 'drop_target_register'):
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.on_drop)

    def _create_layout(self):
        # Frame for Controls (Left)
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # File Operations
        ttk.Label(control_frame, text="File", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        ttk.Button(control_frame, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=5)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # Processing Options
        ttk.Label(control_frame, text="Processing Options", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        
        self.fix_square_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Fix Square (Auto-Crop)", variable=self.fix_square_var).pack(anchor="w", pady=2)
        
        ttk.Label(control_frame, text="Sample Method:").pack(anchor="w", pady=(5, 0))
        self.sample_method_var = tk.StringVar(value="palette-mode")
        sample_combo = ttk.Combobox(control_frame, textvariable=self.sample_method_var, values=["palette-mode", "center", "median", "majority"], state="readonly")
        sample_combo.pack(fill=tk.X, pady=2)

        # --- Grid Options ---
        ttk.Label(control_frame, text="Grid (0 = Auto):").pack(anchor="w", pady=(8, 0))
        grid_row = ttk.Frame(control_frame); grid_row.pack(fill=tk.X, pady=2)
        self.grid_w_var = tk.StringVar(value="0")
        self.grid_h_var = tk.StringVar(value="0")
        ttk.Entry(grid_row, textvariable=self.grid_w_var, width=7).pack(side=tk.LEFT)
        ttk.Label(grid_row, text=" x ").pack(side=tk.LEFT)
        ttk.Entry(grid_row, textvariable=self.grid_h_var, width=7).pack(side=tk.LEFT)
        self.harmonic_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Harmonic Fix (detail rescue)",
                        variable=self.harmonic_var).pack(anchor="w", pady=2)

        ttk.Label(control_frame, text="Refine Intensity:").pack(anchor="w", pady=(5, 0))
        self.refine_var = tk.DoubleVar(value=0.25)
        ttk.Scale(control_frame, from_=0.0, to=0.5, variable=self.refine_var,
                  orient="horizontal").pack(fill=tk.X)

        # --- Palette-mode Options ---
        ttk.Label(control_frame, text="Max Colors (0 = Auto):").pack(anchor="w", pady=(5, 0))
        self.ncolors_var = tk.StringVar(value="0")
        ttk.Entry(control_frame, textvariable=self.ncolors_var, width=7).pack(anchor="w", pady=2)

        ttk.Label(control_frame, text="Color Merge Dist (8-48):").pack(anchor="w", pady=(5, 0))
        self.colordist_var = tk.DoubleVar(value=24.0)
        ttk.Scale(control_frame, from_=8.0, to=48.0, variable=self.colordist_var,
                  orient="horizontal").pack(fill=tk.X)

        ttk.Label(control_frame, text="Cell Margin (0-0.45):").pack(anchor="w", pady=(5, 0))
        self.margin_var = tk.DoubleVar(value=0.25)
        ttk.Scale(control_frame, from_=0.0, to=0.45, variable=self.margin_var,
                  orient="horizontal").pack(fill=tk.X)

        ttk.Button(control_frame, text="Run Perfect Pixel", command=self.process_image).pack(fill=tk.X, pady=10)
        ttk.Button(control_frame, text="View: Original / Result", command=self.toggle_view).pack(fill=tk.X, pady=2)
        
        # New Batch Process Button
        ttk.Button(control_frame, text="Batch Process Multiple Files...", command=self.batch_process).pack(fill=tk.X, pady=5)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # Export Options
        ttk.Label(control_frame, text="Export Options", font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
        
        ttk.Label(control_frame, text="Upscale Factor (Output Size):").pack(anchor="w", pady=(5, 0))
        self.upload_scale_var = tk.StringVar(value="8x")
        scale_values = ["1x (Original)", "2x", "4x", "8x", "16x", "32x", "Custom Width..."]
        self.scale_combo = ttk.Combobox(control_frame, textvariable=self.upload_scale_var, values=scale_values, state="readonly")
        self.scale_combo.pack(fill=tk.X, pady=2)
        self.scale_combo.bind("<<ComboboxSelected>>", self.on_scale_changed)

        self.custom_width_entry = ttk.Entry(control_frame)
        # Hidden by default, shown if Custom Width is selected
        
        ttk.Button(control_frame, text="Save Result", command=self.save_image).pack(fill=tk.X, pady=15)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # Info
        dnd_text = "(Drag & Drop supported)" if HAS_DND else ""
        self.info_label = ttk.Label(control_frame, text=f"Ready.\n{dnd_text}", wraplength=180, justify="left")
        self.info_label.pack(anchor="w", pady=5)

        # Canvas Area (Right)
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#333333", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bindings for Pan and Zoom
        self.canvas.bind("<ButtonPress-2>", self.start_pan) # Middle click press
        self.canvas.bind("<B2-Motion>", self.do_pan)      # Middle click drag
        self.canvas.bind("<MouseWheel>", self.do_zoom)    # Windows scroll
        
        # Also bind Left click for easier panning just in case
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.do_pan)

    def on_drop(self, event):
        data = event.data
        if not data: return
        
        # Regex to handle curly braced paths {path with spaces} or normal paths
        files = []
        if '{' in data:
            files = re.findall(r'\{(.+?)\}', data)
            if not files: files = [data]
        else:
            files = data.split()

        valid_files = [f for f in files if os.path.isfile(f) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
        
        if not valid_files:
            return

        print(f"Dropped files: {valid_files}")
        if len(valid_files) == 1:
            self.load_image(valid_files[0])
        else:
            msg = f"Dropped {len(valid_files)} files.\nDo you want to batch process them?"
            if messagebox.askyesno("Batch Process", msg):
                self.batch_process(file_list=valid_files)
            else:
                self.load_image(valid_files[0]) 

    def on_scale_changed(self, event):
        val = self.upload_scale_var.get()
        if "Custom" in val:
            self.custom_width_entry.pack(fill=tk.X, pady=2, after=self.scale_combo)
            if not self.custom_width_entry.get():
                self.custom_width_entry.insert(0, "1024")
        else:
            self.custom_width_entry.pack_forget()

    def load_image(self, override_path=None):
        if override_path:
            file_path = override_path
        else:
            file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")])
        
        if not file_path:
            return
        
        self.current_image_path = file_path
        try:
            self.original_cv_image = imread_kr(file_path)
            if self.original_cv_image is None:
                raise ValueError("Could not read image.")
            self.original_cv_image = self.original_cv_image.copy()
            self.original_cv_image.setflags(write=False)  # 원본 불변 보장
            self.viewing = "original"
            self.processed_cv_image = None # Reset processed
            self.update_display_image(self.original_cv_image)
            self.info_label.config(text=f"Loaded: {os.path.basename(file_path)}\nSize: {self.original_cv_image.shape[1]}x{self.original_cv_image.shape[0]}")
            
            # Reset view
            self.zoom_scale = 1.0
            self.pan_x = 0
            self.pan_y = 0
            self.update_canvas_view()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")

    def _gather_params(self):
        def _int(v):
            try: return max(0, int(float(v)))
            except Exception: return 0
        return dict(
            method=self.sample_method_var.get(),
            grid_w=_int(self.grid_w_var.get()),
            grid_h=_int(self.grid_h_var.get()),
            harmonic=self.harmonic_var.get(),
            refine_intensity=float(self.refine_var.get()),
            n_colors=_int(self.ncolors_var.get()),
            min_color_dist=float(self.colordist_var.get()),
            margin=float(self.margin_var.get()),
            fix_square=self.fix_square_var.get(),
        )

    def process_image(self):
        if self.original_cv_image is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        method = self.sample_method_var.get()
        fix_sq = self.fix_square_var.get()
        
        try:
            self.root.config(cursor="watch")
            self.root.update()
            
            params = self._gather_params()
            # 항상 '로드 시점의 원본'의 복사본을 처리 (결과물 재처리 원천 차단)
            w, h, out, info = run_pipeline(self.original_cv_image.copy(), **params)

            self.root.config(cursor="")

            if w is None or h is None:
                 messagebox.showerror("Error", f"Perfect Pixel failed.\n{info}")
                 return

            self.processed_cv_image = out
            self.viewing = "result"
            self.update_display_image(self.processed_cv_image)
            src_name = os.path.basename(self.current_image_path) if self.current_image_path else "?"
            self.info_label.config(text=f"Processed from: {src_name}\nGrid: {w}x{h}\nMethod: {params['method']}\n{info}")
            
            # Auto zoom in for small pixel art
            if w < 100:
                self.zoom_scale = max(1.0, 500 / max(w, h))
            
            self.pan_x = 0
            self.pan_y = 0
            self.update_canvas_view()

        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("Error", f"Processing failed:\n{e}")

    def batch_process(self, file_list=None):
        # 1. Select Files
        if file_list:
            file_paths = file_list
        else:
            file_paths = filedialog.askopenfilenames(title="Select Images to Process", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")])
        
        if not file_paths:
            return

        # 2. Select Output Directory
        output_dir = filedialog.askdirectory(title="Select Output Folder")
        if not output_dir:
            return

        # 3. Get Settings
        params = self._gather_params()
        
        success_count = 0
        fail_count = 0
        
        self.root.config(cursor="watch")
        
        progress_popup = tk.Toplevel(self.root)
        progress_popup.title("Batch Processing...")
        progress_popup.geometry("300x120")
        
        lbl = tk.Label(progress_popup, text="Processing...", pady=10)
        lbl.pack()
        progress_bar = ttk.Progressbar(progress_popup, length=250, mode='determinate')
        progress_bar.pack(pady=5)
        progress_bar['maximum'] = len(file_paths)

        try:
            for idx, file_path in enumerate(file_paths):
                progress_bar['value'] = idx + 1
                lbl.config(text=f"Processing {idx+1}/{len(file_paths)}: {os.path.basename(file_path)}")
                progress_popup.update()
                
                try:
                    # Read (한글 경로 안전)
                    rgb = imread_kr(file_path)
                    if rgb is None: raise ValueError("Read Error")

                    # Process
                    w, h, out, info = run_pipeline(rgb, **params)

                    if w is None or h is None:
                        fail_count += 1
                        continue
                        
                    # Upscale
                    final_img = self._get_upscaled_image(out)
                    
                    # Save
                    if final_img is not None:
                        base_name = os.path.basename(file_path)
                        name, ext = os.path.splitext(base_name)
                        save_name = f"{name}_pixelized{ext}" if ext else f"{name}_pixelized.png"
                        save_path = os.path.join(output_dir, save_name)
                        
                        imwrite_kr(save_path, final_img)
                        success_count += 1
                    else:
                        fail_count += 1

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    fail_count += 1
        finally:
            self.root.config(cursor="")
            progress_popup.destroy()
            messagebox.showinfo("Batch Complete", f"Processed {success_count} images.\nFailed: {fail_count}")

    def _get_upscaled_image(self, cv_img):
        """Helper to resize image based on current GUI settings."""
        if cv_img is None: return None
        
        scale_opt = self.upload_scale_var.get()
        h, w = cv_img.shape[:2]
        target_w, target_h = w, h

        try:
            if "Custom" in scale_opt:
                val = self.custom_width_entry.get()
                if not val.isdigit():
                     return None # Or raise error
                target_w_req = int(val)
                # Maintain aspect ratio
                ratio = target_w_req / w
                target_w = target_w_req
                target_h = int(h * ratio)
            elif "x" in scale_opt: # 2x, 4x, etc
                factor_str = scale_opt.split("x")[0]
                if factor_str.isdigit():
                    factor = int(factor_str)
                    target_w = w * factor
                    target_h = h * factor
            
            # Upscale
            return cv2.resize(cv_img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        except:
            return None

    def update_display_image(self, cv_img):
        # Convert CV image (RGB) to PIL Image
        self.display_pil_image = Image.fromarray(cv_img)

    def update_canvas_view(self):
        if not hasattr(self, 'display_pil_image'):
            return
            
        # Calculate new size based on zoom
        orig_w, orig_h = self.display_pil_image.size
        new_w = int(orig_w * self.zoom_scale)
        new_h = int(orig_h * self.zoom_scale)
        
        # Don't try to resize to 0
        if new_w <= 0 or new_h <= 0:
            return

        # Resize for display (Nearest Neighbor for crisp pixels)
        resized = self.display_pil_image.resize((new_w, new_h), Image.Resampling.NEAREST)
        self.tk_image = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        
        # Center the image plus pan offset
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        center_x = canvas_w // 2 + self.pan_x
        center_y = canvas_h // 2 + self.pan_y
        
        self.canvas_image_id = self.canvas.create_image(center_x, center_y, image=self.tk_image, anchor="center")

    def start_pan(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def do_pan(self, event):
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        self.pan_x += dx
        self.pan_y += dy
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.update_canvas_view()

    def do_zoom(self, event):
        # Determine zoom direction
        if event.delta > 0:
            factor = 1.1
        else:
            factor = 0.9
            
        new_zoom = self.zoom_scale * factor
        # Cap zoom
        if 0.1 < new_zoom < 50.0:
            self.zoom_scale = new_zoom
            self.update_canvas_view()

    def toggle_view(self):
        """원본 ↔ 처리 결과 보기 전환 (현재 무엇을 보고 있는지 확인용)."""
        if self.original_cv_image is None:
            return
        if getattr(self, "viewing", "original") == "result" or self.processed_cv_image is None:
            self.viewing = "original"
            self.update_display_image(self.original_cv_image)
            h, w = self.original_cv_image.shape[:2]
            label = f"Viewing: ORIGINAL ({w}x{h})"
        else:
            self.viewing = "result"
            self.update_display_image(self.processed_cv_image)
            h, w = self.processed_cv_image.shape[:2]
            label = f"Viewing: RESULT ({w}x{h})"
        self.info_label.config(text=label)
        self.update_canvas_view()

    def save_image(self):
        if self.processed_cv_image is None:
             messagebox.showwarning("Warning", "No processed image to save. Run 'Perfect Pixel' first.")
             return

        # 기본 경로 = 원본 이미지 폴더, 기본 파일명 = 원본이름_시각
        init_dir = ""
        base = "result"
        if self.current_image_path:
            init_dir = os.path.dirname(self.current_image_path)
            base = os.path.splitext(os.path.basename(self.current_image_path))[0]
        default_name = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialdir=init_dir,
            initialfile=default_name,
            filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")])
        if not file_path:
            return

        try:
            final_img = self._get_upscaled_image(self.processed_cv_image)
            if final_img is None:
                 messagebox.showerror("Error", "Failed to resize image. Check settings.")
                 return

            imwrite_kr(file_path, final_img)
            
            h, w = final_img.shape[:2]
            messagebox.showinfo("Success", f"Saved to {file_path}\nSize: {w}x{h}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image:\n{e}")

if __name__ == "__main__":
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    app = PerfectPixelGUI(root)
    # Force initial update to get canvas geometry for centering
    root.update() 
    root.mainloop()
