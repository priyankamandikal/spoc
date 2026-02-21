"""
Render overlayed videos from existing frames + segmentation masks.

Example:
  python eval/render_overlay_videos.py --dataset WTC-HowTo --verb chopping --out-fps 2 --nproc 10
  python eval/render_overlay_videos.py --dataset WTC-VOST --verb cut --out-fps 2 --nproc 10

Expected per-video layout:
  frames: <datadir>/<osc>/JPEGImages_1fps/<video_name>/frame*.jpg (or *.jpg)
  masks : <datadir>/<osc>/gt/masks/<video_name>/frame*.png
  out   : <datadir>/<osc>/gt/videos/<video_name>.mp4
"""

import os
import os.path as osp
import multiprocessing as mp
import cv2
import numpy as np
from PIL import Image as PILImage

from myutils.viz_utils import show_anns_osc, init_ffmpeg_process, close_ffmpeg_process


def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--verb", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True, help="Dataset name. Choose from: WTC-HowTo, WTC-VOST")
    p.add_argument("--split", type=str, default="eval", help="Data split. Choose from: train, eval")
    p.add_argument("--out-fps", type=float, default=2.0)
    p.add_argument("--nproc", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--noun", type=str, default=None, help="Optional noun filter. Combined with --verb as <verb>_<noun>.")
    p.add_argument("--video-name", type=str, default=None, help="Optional single video_name filter.")
    return p.parse_args()


def list_frames(frames_dir: str):
    if not osp.isdir(frames_dir):
        return []
    # Accept common jpg naming patterns
    frames = [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    frames.sort()
    return [osp.join(frames_dir, f) for f in frames]


def load_mask_png(mask_path: str) -> np.ndarray:
    # Keep label indices (0/1/2). Palettized PNGs load fine; convert("P") is safe.
    return np.array(PILImage.open(mask_path).convert("P"), dtype=np.uint8)


def overlay_one(frame_rgb: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    # mask values: 0/1/2 for background/actionable/transformed
    # show_anns_osc expects list of boolean masks + labels
    masks = []
    labels = []

    if (mask == 1).any():
        masks.append(mask == 1)
        labels.append(["pre"])
    if (mask == 2).any():
        masks.append(mask == 2)
        labels.append(["post"])

    if len(masks) == 0:
        return frame_rgb

    out = show_anns_osc(tuple(masks), tuple(labels), alpha=alpha, frame=frame_rgb, fname=None, mode="return")
    return out


def process_video(video_id: str, args, tmp_logfile: str):
    """
    video_id is expected like: "<osc>/<video_name>"
    where osc is e.g. chopping_avocado, video_name is e.g. 0YjWzXseLTY_st0.0_dur40.0
    """
    try:
        osc, video_name = video_id.split("/", 1)
    except ValueError:
        with open(tmp_logfile, "a") as f:
            f.write(f"{video_id}: ERROR: expected 'osc/video_name'\n")
        return

    frames_dir = osp.join(args.datadir, osc, "JPEGImages_1fps", video_name)
    masks_dir  = osp.join(args.datadir,  osc, "gt/masks", video_name)

    frame_paths = list_frames(frames_dir)
    if not frame_paths:
        with open(tmp_logfile, "a") as f:
            f.write(f"{video_id}: WARN: no frames at {frames_dir}\n")
        return

    # Match masks directly by frame basename.
    aligned_pairs = []
    for frame_path in frame_paths:
        stem = osp.splitext(osp.basename(frame_path))[0]
        mp_aligned = osp.join(masks_dir, f"{stem}.png")
        if osp.isfile(mp_aligned):
            aligned_pairs.append((frame_path, mp_aligned))

    if len(aligned_pairs) == 0:
        with open(tmp_logfile, "a") as f:
            f.write(f"{video_id}: WARN: no masks at {masks_dir}\n")
        return

    n = len(aligned_pairs)

    out_video = osp.join(args.datadir, osc, "gt/videos", f"{video_name}.mp4")
    os.makedirs(osp.dirname(out_video), exist_ok=True)
    if osp.exists(out_video):
        os.remove(out_video)

    # Initialize writer from first frame
    frame0 = cv2.imread(frame_paths[0])
    if frame0 is None:
        with open(tmp_logfile, "a") as f:
            f.write(f"{video_id}: ERROR: failed to read {frame_paths[0]}\n")
        return

    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    h, w, _ = frame0.shape
    ffmpeg_process = init_ffmpeg_process(out_video, w, h, args.out_fps)

    # Process frames
    for i in range(n):
        fr_path, mpth = aligned_pairs[i]
        fr = cv2.imread(fr_path)
        if fr is None:
            continue
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        mask = load_mask_png(mpth)

        over = overlay_one(fr, mask, alpha=args.alpha)
        ffmpeg_process.stdin.write(cv2.cvtColor(over, cv2.COLOR_BGR2RGB).astype(np.uint8).tobytes())

    close_ffmpeg_process(ffmpeg_process)

    with open(tmp_logfile, "a") as f:
        f.write(f"{video_id}: OK -> {out_video} (frames={n})\n")


def worker(video_batch, pid, args, logdir):
    tmp_logfile = osp.join(logdir, f"render_{pid}.txt")
    for vid in video_batch:
        print(f"[pid {pid}] {vid}")
        process_video(vid, args, tmp_logfile)


def main():
    args = parse_args()
    args.datadir = osp.join("data/WhereToChange/eval", args.dataset)
    logdir = "logs/render-overlays"
    os.makedirs(logdir, exist_ok=True)

    split_file = osp.join("data/WhereToChange/metadata", args.dataset, "subset", args.verb, f"{args.split}.txt")
    with open(split_file, "r") as f:
        videos = [line.strip() for line in f if line.strip()]

    # Optional filtering for targeted rendering.
    if args.noun is not None and args.noun != "":
        osc = f"{args.verb}_{args.noun}"
        videos = [v for v in videos if v.startswith(f"{osc}/")]
    if args.video_name is not None and args.video_name != "":
        videos = [v for v in videos if v.endswith(f"/{args.video_name}")]

    total_processes = min(args.nproc, max(1, len(videos)))
    print(f"Total videos: {len(videos)}")
    print(f"Spawning {total_processes} processes")

    batches = [videos[i::total_processes] for i in range(total_processes)]
    procs = []
    for pid, batch in enumerate(batches):
        p = mp.Process(target=worker, args=(batch, pid, args, logdir))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
