# SPOC

## WhereToChange Eval

- The WhereToChange eval set can be downloaded from [here](#).
    - Please unzip and move it to `spoc/data`
    - It contains two subsets -- WTC-HowTo and WTC-VOST
    - Video clips as well as images and masks annotated at 1fps are provided
- Expected dataset structure:
```text
spoc/
└── data/
    └── WhereToChange/
        ├── eval/
        │   ├── WTC-HowTo/
        │   │   └── <osc>/
        │   │       ├── clips/
        │   │       ├── JPEGImages_1fps/
        │   │       └── gt/
        │   └── WTC-VOST/
        │       └── ...
        └── metadata/
```
- The annotated video can be visualized by running `eval/render_overlay_videos.py`.
  - Example:
    `python eval/render_overlay_videos.py --verb chopping --dataset WTC-HowTo --split eval --fps 2 --nproc 10 --noun avocado --video-name 8qq_gQnbTRU_st205.0_dur40.0`
  - You can remove `--video-name` and `--noun` to generate annotated videos for all videos in a verb.
