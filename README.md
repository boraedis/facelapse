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
curl -L -o models/face_recognition_sface_2021dec.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx
brew install ffmpeg
mkdir -p work/me && cp ~/path/to/a-clear-recent-photo-of-you.jpg work/me/
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
another folder. `scan` takes several folders at once.

Dates come from, in order: a Takeout `.json` file, the photo's own embedded
date, or the filename (`20090315_…`, or Photo Booth's `Photo on 9-25-17 at 12.42 PM`).
For anything still undated, add a line to `work/dates.txt`:

```
IMG_4890.JPG            2017-06
IMG_2455.heic           2019
```

Mac Photo Booth saves mirror images. Files it named (`Photo on …`, `4-up on …`)
are flipped back automatically, so your face doesn't swap sides between those
frames and camera frames.

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
- **select** first works out which face in each photo is you. It starts
  from the photos in `work/me/`, then learns more of your faces by linking
  through similar-looking photos, so childhood photos get recognised from an
  adult reference. It then removes photos with someone else inside the
  cropped frame, turned or tilted heads, closed eyes, tiny faces, and faces
  too close to the photo's edge.
- By default every usable photo becomes a frame, in date order. Shots taken
  in the same minute (bursts, duplicate copies) collapse to the best one. For
  an even pace through time, use `--period week|month|quarter|year` instead
  to keep only the best photo per period.
- **review.html** shows each month's frames (green), already cropped the
  way the final frame will be, with that month's rejected photos (uncropped,
  with the reason) folded underneath, and undated photos at the end. The pin and exclude
  buttons collect file paths into the boxes at the top. Paste them into
  `work/pin.txt` or `work/exclude.txt` and run `select` again. Both files
  persist, so this year's choices carry over to next year's run.
- Framing (`--width/--height/--eye-y/--eye-dist`) must be the same for
  `select` and `align`.
- **align** also evens out lighting and colour: each frame's face brightness
  (gamma, so backgrounds don't blow out) and colour cast are pulled toward the
  median frame. `--color 0` turns it off, `--color 1` matches fully (default 0.8).
- **encode** plays 12 photos a second by default (`--fps`).
- **polaroid** renders the landing-page hero version: each photo drops onto a
  growing pile of polaroids, eye-aligned, speeding up through the middle
  (`--edge-rate`/`--peak-rate`) and landing on the latest photo. Cards show
  only the real part of each photo around the face (`--eye-dist` sets the
  zoom), so nothing is filled in. `--bg light|dark` matches the site's two
  themes; writes `work/polaroid-<bg>.mp4` and a poster of the finished pile.
  Run `align` first.
- Turning heads toward the camera was tried and dropped: warping a 2D photo
  around a 3D pose visibly distorted eyes and cheeks, even at partial strength.
- The filter thresholds are constants at the top of `facelapse.py`.

## Checked on synthetic data

Tested on copies of one portrait that were rotated (−9° to +12°), scaled
(0.6× to 1.3×), stored sideways with an EXIF orientation tag, and dated
through a Takeout JSON file with a truncated name. On every output frame, the
re-detected eyes landed within 2.4px of the target on a 1080px canvas. The
undated copy, the blank image, and the 0.6× copy (face too small) were
rejected.

On a real set of 179 photos (2004–2026, mostly 2016–18 Photo Booth shots,
many with a friend in the frame), the right person was picked in every
frame on the contact sheet, including a 2005 childhood photo.
