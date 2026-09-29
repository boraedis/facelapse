# facelapse

Turns a folder of photos of one person into an eye-aligned timelapse video.
It was built for the data-diary landing page (boraedis/data-diary#457, epic #15).
It's an offline tool you run about once a year. It's deliberately kept out of
the app itself.

## Setup (once)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p models && curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
brew install ffmpeg
```

## Getting the photos out of Google Photos

1. Google Photos → **Search → People** → open your own face group.
2. Select the photos and **Add to album** (for example "facelapse"). Shift-click
   selects a date range.
3. Go to [Google Takeout](https://takeout.google.com), deselect everything,
   select **Google Photos → only the "facelapse" album**, and export.
4. Unzip it anywhere outside this repo. Keep the `.json` files next to the
   photos. They hold the date each photo was taken, which survives even when
   the embedded photo date has been stripped.

Older photos from other places (for example high-school ones) can simply be
another folder. `scan` takes several folders at once. Undated photos need a
date in the filename (`20090315_anything.jpg`).

## Running it

```bash
PY=.venv/bin/python
$PY facelapse.py scan ~/Downloads/Takeout/Google\ Photos/facelapse [more folders…]
$PY facelapse.py select            # writes work/review.html, open it in a browser
$PY facelapse.py align
$PY facelapse.py encode --blend    # work/facelapse.mp4 + poster.jpg/.webp
```

- **scan** caches its results in `work/scan.jsonl`. It can be resumed after a
  Ctrl-C, and next year it only processes the new photos.
- **select** removes group shots, turned or tilted heads, closed eyes, tiny
  faces, and faces too close to the photo's edge. It then scores the rest
  (how front-facing, sharpness, resolution, eyes open) and keeps the best
  photo per month (`--period week|month|quarter|year`).
- **review.html** shows each period's pick (green) next to its runners-up,
  already cropped the way the final frame will be. The pin and exclude
  buttons collect file paths into the boxes at the top. Paste them into
  `work/pin.txt` or `work/exclude.txt` and run `select` again. Both files
  persist, so this year's choices carry over to next year's run.
- Framing (`--width/--height/--eye-y/--eye-dist`) must be the same for
  `select` and `align`.
- The filter thresholds are constants at the top of `facelapse.py`.

## Checked on synthetic data

Tested on copies of one portrait that were rotated (−9° to +12°), scaled
(0.6× to 1.3×), stored sideways with an EXIF orientation tag, and dated
through a Takeout JSON file with a truncated name. On every output frame, the
re-detected eyes landed within 2.4px of the target on a 1080px canvas. The
undated copy, the blank image, and the 0.6× copy (face too small) were
rejected. The group-photo filter hasn't been tested yet.
