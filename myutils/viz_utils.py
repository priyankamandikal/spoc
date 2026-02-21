'''
Utility functions for visualizing images and masks.
'''

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import ffmpeg
import os.path as osp
from scipy.ndimage import binary_dilation

np.random.seed(200)
_palette = ((np.random.random((3*255))*0.7+0.3)*255).astype(np.uint8).tolist()
_palette = [0,0,0]+_palette

osc_palette = [
    0, 0, 0,        # Black background
    255, 102, 102,  # Red for pre
    153, 255, 153,  # Green for post
    153, 204, 255,  # Blue for bg
    255, 255, 153,  # Yellow for amb
    204, 204, 204,  # Grey for ignore
    ]

osc_labels = {
    "pre": 1,
    "post": 2,
    "bg": 3,
    "amb": 4,
    "ignore": 5
}

def colorize_mask(pred_mask):
    """Convert a prediction mask into a color image using a predefined palette."""
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(osc_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)

def draw_mask(img, mask, alpha=0.7, id_contour=False):
    """Overlay a colorized mask onto an image with optional contours."""
    binary_mask = (mask != 0)
    contours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
    foreground = img * (1 - alpha) + colorize_mask(mask) * alpha
    img_mask = img.copy()
    img_mask[binary_mask] = foreground[binary_mask]
    if id_contour:
        img_mask[contours, :] = 0
    return img_mask.astype(img.dtype)

def show_anns_osc(masks, labels, confs=None, alpha=0.35, ax=None, fname=None, frame=None, mode='show', save_path=None):
    # aggr_mask_1 = np.zeros_like(masks[0], dtype=np.uint8)
    aggr_mask = np.zeros_like(masks[0], dtype=np.uint8)
    if confs is None:
        for m, l in zip(masks, labels):
            aggr_mask[m] = osc_labels[l[0]]
    else:
        conf_map = np.full_like(masks[0], fill_value=-np.inf, dtype=np.float32)
        for m, l, c in zip(masks, labels, confs):
            mask_positions = (m != 0)
            update_positions = mask_positions & (c > conf_map)
            aggr_mask[update_positions] = osc_labels[l[0]]
            conf_map[update_positions] = c
    img_mask = draw_mask(frame, aggr_mask, alpha=alpha, id_contour=True)
    if fname is not None:
        display_x = 0.05 * img_mask.shape[1]
        display_y = 0.05 * img_mask.shape[0]
        cv2.putText(img_mask, fname, (int(display_x), int(display_y)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2, cv2.LINE_AA)
    if mode == 'save':
        assert save_path is not None, "save_path must be provided for save mode."
        cv2.imwrite(save_path, cv2.cvtColor(img_mask, cv2.COLOR_RGB2BGR))
        return None
    elif mode == 'show':
        plt.imshow(img_mask)
        plt.axis('off')
        plt.show()
    elif mode == 'return':
        return cv2.cvtColor(img_mask, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
def save_samtrack_mask(pred_mask,output_dir,file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(osp.join(output_dir,file_name))

def init_ffmpeg_process(video_out_path, width, height, fps):
    ffmpeg_process = (
                    ffmpeg
                    .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', r=fps)
                    .output(video_out_path, vcodec='libx264', pix_fmt='yuv420p', r=fps, loglevel='quiet')
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
                )
    return ffmpeg_process

def close_ffmpeg_process(ffmpeg_process):
    if ffmpeg_process:
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()