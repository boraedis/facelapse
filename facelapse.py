#!/usr/bin/env python3
"""Facelapse: turn a pile of photos of one person into an eye-aligned timelapse.

Built for data-diary#457 (the landing-page facelapse, epic #15). Deliberately
a standalone offline tool, not part of the data-diary app: it runs maybe once
a year, on a laptop, against a local folder of exported photos.

Pipeline (each step reads the previous step's output from --work):

  scan    Walk one or more photo folders, date every photo, detect faces with
          MediaPipe, and fingerprint each face (OpenCV SFace) so `select` can
          tell you apart from friends. Cached in scan.jsonl; a re-run only
          processes new or changed files.
  select  Work out which face in each photo is you, filter out unusable shots
          (someone else in frame, turned heads, closed eyes, tiny faces), and
          keep every usable photo in date order (or the best per period).
          Writes selection.csv plus review.html, a contact sheet of picks and
          rejects.
  align   Re-detect eyes precisely on each selected photo and warp it so the
          eyes land on fixed canvas coordinates. Writes frames/ + frames.csv.
  encode  ffmpeg the frames into an MP4 (optionally WebM) plus a poster image
          of the final frame.

Files you maintain in --work, all honoured by `select` on every run so choices
survive a yearly re-run:

  me/           one or more clear, recent photos of you: the identity seed
  exclude.txt   never use these photos (one path or file name per line)
  pin.txt       always use these photos, overriding the quality filters
  dates.txt     `<file name or path>  YYYY[-MM[-DD]]` for photos with no date

Alignment is eyes-only (decision 5 on #15): mouth position moves with
expression, so pinning it as a second anchor fights the eye alignment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pillow_heif
from PIL import Image, ImageOps

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# Google Photos exports iPhone shots as HEIC; Pillow can't open them alone.
pillow_heif.register_heif_opener()

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "face_landmarker.task"
SFACE_PATH = ROOT / "models" / "face_recognition_sface_2021dec.onnx"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}

# Bump when scan's output changes shape or meaning; older cache lines are
# rescanned. v2: face embeddings, Photo Booth dates, Photo Booth un-mirroring.
SCAN_VERSION = 2

# Detection runs on a downscaled copy: MediaPipe's detector works at a few
# hundred px internally anyway, and decoding + scanning thousands of 12MP
# photos at full size would make `scan` several times slower for no gain.
# `align` re-detects on a full-resolution face crop, which is where precision
# actually matters.
DETECT_MAX_SIDE = 1600
REFINE_CROP_SIDE = 768

# MediaPipe face-mesh landmark indices. Eye centres are the midpoint of each
# eye's two corners rather than the iris, because the iris moves with gaze
# and would make the head drift whenever the subject looks sideways.
EYE_A_CORNERS = (33, 133)   # subject's right eye (image left)
EYE_B_CORNERS = (362, 263)  # subject's left eye (image right)
NOSE_TIP = 1
MOUTH_A = 61                # mouth corner, image left
MOUTH_B = 291               # mouth corner, image right
CHEEK_A = 234
CHEEK_B = 454

# Mac Photo Booth saves mirror images by default ("Auto Flip New Items"), so
# its shots are flipped back on load. Left alone, the face would swap sides
# (hair parting, asymmetries) every time the video cut between a Photo Booth
# frame and a camera frame. Matched on Photo Booth's own file naming.
PHOTO_BOOTH = re.compile(r"^(Photo|4-up|Movie) on \d{1,2}-\d{1,2}-\d{2,4} at ")

# --- Identity (select). ---
# OpenCV's published SFace cosine threshold for "same person".
MATCH_MIN = 0.363
# Stricter bar for a face to join the "this is me" set that other photos are
# compared against. The set starts from work/me/ and grows by chaining
# through similar-looking photos, which is how a 2004 face gets recognised
# from a 2026 reference: each hop only has to bridge a few years. A strict
# bar keeps a lookalike friend from joining and dragging the chain off.
JOIN_MIN = 0.45
# In a photo with several faces, the best match must beat the runner-up by
# this much, or the photo is ambiguous rather than a match.
MATCH_MARGIN = 0.08

# --- Filter thresholds (select). Tune here after looking at review.html. ---
# Loosened on #457 after the first real run: in a fast timelapse a slightly
# turned or tilted face flashes by fine, and more photos (especially the
# sparse early years and a recent final frame) beat stricter quality.
# Another face only disqualifies a shot if it's at least this fraction of
# your face's size (so distant strangers don't count) and lands inside the
# output frame, with this much margin (fraction of canvas width) around it
# for a face that would be half-visible at the edge.
OTHER_FACE_RATIO = 0.45
OTHER_FACE_MARGIN = 0.1
# Minimum eye-centre distance in source pixels. Below this the face has to be
# upscaled so far into the canvas that it reads as mush.
MIN_EYE_DIST_PX = 45
# Head-turn proxy in [-1, 1]: nose-to-cheek asymmetry along the eye line.
# ~0 is frontal; 0.2 is a noticeable three-quarter turn, 0.45 about as far
# as still reads as a face-on frame (true profiles score 0.8+).
MAX_YAW = 0.45
# Head tilt in degrees. Alignment removes roll entirely, but a heavily tilted
# head usually means a lying-down or goofy shot that looks off once levelled.
MAX_ROLL_DEG = 40
# MediaPipe blendshape eyeBlink score (0 open .. 1 closed). Squints score
# ~0.3-0.4, so this rejects only genuinely closed eyes.
MAX_BLINK = 0.65
# Fraction of the output canvas the source photo must cover once aligned.
# Faces near a photo's edge leave a gap that gets filled with smeared edge
# pixels; a sliver is fine, a quarter of the frame is not.
MIN_COVERAGE = 0.75


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_landmarker(num_faces: int) -> vision.FaceLandmarker:
    if not MODEL_PATH.exists():
        sys.exit(f"Missing model: {MODEL_PATH}\nSee README.md → Setup.")
    options = vision.FaceLandmarkerOptions(
        # CPU explicitly: on macOS MediaPipe otherwise tries a Metal helper
        # that aborts the whole process when the GPU service is unavailable
        # (sandboxes, some remote sessions). CPU manages several photos a
        # second, and the bottleneck is decoding the JPEG/HEIC, not inference.
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH), delegate=BaseOptions.Delegate.CPU),
        num_faces=num_faces,
        output_face_blendshapes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


def load_recognizer():
    if not SFACE_PATH.exists():
        sys.exit(f"Missing model: {SFACE_PATH}\nSee README.md → Setup.")
    return cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")


def load_image(path: Path) -> tuple[Image.Image, Image.Image]:
    """Return (raw, upright RGB). The raw image keeps its EXIF for dating.

    exif_transpose matters: phone photos are usually stored sideways with an
    orientation tag, and every coordinate this tool records is in upright
    space, so detection and warping must both see the same upright pixels.
    Photo Booth shots are un-mirrored here for the same reason.
    """
    raw = Image.open(path)
    upright = ImageOps.exif_transpose(raw).convert("RGB")
    if PHOTO_BOOTH.match(path.name):
        upright = ImageOps.mirror(upright)
    return raw, upright


def detect(landmarker: vision.FaceLandmarker, rgb: np.ndarray):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    return landmarker.detect(image)


def face_points(landmarks, width: int, height: int) -> np.ndarray:
    return np.array([(p.x * width, p.y * height) for p in landmarks], dtype=np.float64)


def eye_centres(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = pts[list(EYE_A_CORNERS)].mean(axis=0)
    b = pts[list(EYE_B_CORNERS)].mean(axis=0)
    return a, b


def embed(recognizer, bgr: np.ndarray, pts: np.ndarray) -> list[float]:
    """Unit-length SFace identity vector for one face.

    SFace's alignCrop expects YuNet's detection row (box + 5 landmarks, eyes
    and mouth corners in image-left-first order); MediaPipe's mesh has all
    five, so the row is built from it rather than running a second detector.
    """
    eye_a, eye_b = eye_centres(pts)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    row = np.array([[x0, y0, x1 - x0, y1 - y0, *eye_a, *eye_b, *pts[NOSE_TIP],
                     *pts[MOUTH_A], *pts[MOUTH_B], 1.0]], dtype=np.float32)
    feature = recognizer.feature(recognizer.alignCrop(bgr, row)).flatten()
    feature /= np.linalg.norm(feature)
    return [round(float(v), 4) for v in feature]


def similarity(src_a, src_b, dst_a, dst_b) -> np.ndarray:
    """2x3 rotate+scale+translate matrix mapping src eye points onto dst's."""
    s = np.asarray(src_b) - np.asarray(src_a)
    d = np.asarray(dst_b) - np.asarray(dst_a)
    scale = np.hypot(*d) / np.hypot(*s)
    angle = math.atan2(d[1], d[0]) - math.atan2(s[1], s[0])
    a, b = scale * math.cos(angle), scale * math.sin(angle)
    tx = dst_a[0] - (a * src_a[0] - b * src_a[1])
    ty = dst_a[1] - (b * src_a[0] + a * src_a[1])
    return np.array([[a, -b, tx], [b, a, ty]])


class Framing:
    """Where the eyes land on the output canvas. Shared by select + align so
    the review previews and coverage scores match the real frames."""

    def __init__(self, width: int, height: int, eye_y: float, eye_dist: float):
        self.width, self.height = width, height
        half = eye_dist * width / 2
        cx, cy = width / 2, eye_y * height
        self.eye_a = (cx - half, cy)
        self.eye_b = (cx + half, cy)
        self.eye_px = eye_dist * width

    def matrix(self, eye_a, eye_b) -> np.ndarray:
        return similarity(eye_a, eye_b, self.eye_a, self.eye_b)

    def coverage(self, eye_a, eye_b, src_w: int, src_h: int) -> float:
        m = self.matrix(eye_a, eye_b)
        corners = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_h]], dtype=np.float64)
        mapped = (corners @ m[:, :2].T + m[:, 2]).astype(np.float32)
        canvas = np.array(
            [[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]],
            dtype=np.float32,
        )
        area, _ = cv2.intersectConvexConvex(mapped, canvas)
        return float(area) / (self.width * self.height)

    def in_frame(self, m: np.ndarray, point, margin: float) -> bool:
        x, y = m[:, :2] @ np.asarray(point, float) + m[:, 2]
        pad = margin * self.width
        return -pad <= x <= self.width + pad and -pad <= y <= self.height + pad

    def warp(self, rgb: np.ndarray, eye_a, eye_b, scale: float = 1.0) -> np.ndarray:
        """Warp an upright RGB array. `scale` shrinks the canvas for previews.

        When the photo is being shrunk (face bigger than the canvas slot),
        pre-downscale with INTER_AREA first: warpAffine has no area filter and
        would alias a 4000px photo straight down to 1080.
        """
        eye_a, eye_b = np.asarray(eye_a, float), np.asarray(eye_b, float)
        target = self.eye_px * scale
        shrink = target / np.hypot(*(eye_b - eye_a))
        if shrink < 1:
            h, w = rgb.shape[:2]
            rgb = cv2.resize(
                rgb, (max(1, round(w * shrink)), max(1, round(h * shrink))),
                interpolation=cv2.INTER_AREA,
            )
            eye_a, eye_b = eye_a * shrink, eye_b * shrink
        m = self.matrix(eye_a, eye_b)
        m[:, :] *= scale
        size = (round(self.width * scale), round(self.height * scale))
        # BORDER_REPLICATE, not black: select already rejects shots that
        # leave a big gap, and a smeared sliver reads far better than a hard
        # black wedge flickering in and out between frames.
        return cv2.warpAffine(
            rgb, m, size, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
        )


def add_framing_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--width", type=int, default=1080)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--eye-y", type=float, default=0.42,
                   help="eye line height as a fraction of canvas height")
    p.add_argument("--eye-dist", type=float, default=0.20,
                   help="eye-centre spacing as a fraction of canvas width")


def framing_from(args) -> Framing:
    return Framing(args.width, args.height, args.eye_y, args.eye_dist)


# ---------------------------------------------------------------------------
# Dating
# ---------------------------------------------------------------------------

DUP_SUFFIX = re.compile(r"\(\d+\)$")
EDITED_SUFFIX = re.compile(r"-(edited|bearbeitet|modifié|editado)$", re.I)
FILENAME_DATE = re.compile(r"(19|20)(\d{2})[-_]?(\d{2})[-_]?(\d{2})")
# Photo Booth: "Photo on 9-25-17 at 12.42 PM #2", US month-day-year.
PHOTO_BOOTH_DATE = re.compile(
    r" on (\d{1,2})-(\d{1,2})-(\d{2}|\d{4}) at (\d{1,2})\.(\d{2})(?:\.(\d{2}))? ?(AM|PM)", re.I
)


class SidecarIndex:
    """Google Takeout's photoTakenTime, keyed by original filename.

    Takeout's sidecar naming is a mess (`x.jpg.json`,
    `x.jpg.supplemental-metadata.json`, both truncated at ~46 chars, and
    `x(1).jpg` pairing with `x.jpg(1).json`). Rather than reverse that, read
    every JSON's own `title` field (the original filename) once per folder
    and match images against it.
    """

    def __init__(self):
        self._dirs: dict[Path, dict[str, int]] = {}

    def _load(self, folder: Path) -> dict[str, int]:
        if folder not in self._dirs:
            titles: dict[str, int] = {}
            for js in folder.glob("*.json"):
                try:
                    data = json.loads(js.read_text(encoding="utf-8"))
                    ts = int(data["photoTakenTime"]["timestamp"])
                    titles[data["title"]] = ts
                except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError):
                    continue
            self._dirs[folder] = titles
        return self._dirs[folder]

    def lookup(self, path: Path) -> int | None:
        titles = self._load(path.parent)
        if not titles:
            return None
        stem, ext = path.stem, path.suffix
        for candidate in (
            path.name,
            DUP_SUFFIX.sub("", stem) + ext,
            EDITED_SUFFIX.sub("", stem) + ext,
        ):
            if candidate in titles:
                return titles[candidate]
        # Long names get truncated on disk but not in `title`.
        if len(stem) >= 30:
            for title, ts in titles.items():
                if title.startswith(stem):
                    return ts
        return None


def exif_datetime(raw: Image.Image) -> datetime | None:
    try:
        exif = raw.getexif()
        value = exif.get_ifd(0x8769).get(36867) or exif.get(306)  # DateTimeOriginal, DateTime
        if value:
            return datetime.strptime(str(value).strip()[:19], "%Y:%m:%d %H:%M:%S")
    except (ValueError, KeyError, AttributeError):
        pass
    return None


def filename_datetime(path: Path) -> datetime | None:
    m = PHOTO_BOOTH_DATE.search(path.stem)
    if m:
        month, day, year, hour, minute, second, ampm = m.groups()
        year = int(year) + (2000 if len(year) == 2 else 0)
        hour = int(hour) % 12 + (12 if ampm.upper() == "PM" else 0)
        try:
            return datetime(year, int(month), int(day), hour, int(minute), int(second or 0))
        except ValueError:
            return None
    m = FILENAME_DATE.search(path.stem)
    if not m:
        return None
    try:
        return datetime(int(m.group(1) + m.group(2)), int(m.group(3)), int(m.group(4)))
    except ValueError:
        return None


def date_photo(path: Path, raw: Image.Image, sidecars: SidecarIndex) -> tuple[str | None, str]:
    """Taken date as ISO string, plus which source it came from.

    Order: Takeout sidecar (Google's own record, survives EXIF stripping by
    messaging apps), then EXIF, then a date in the filename. File mtime is
    never used: after a download or copy it's the copy time, not the shot.
    """
    ts = sidecars.lookup(path)
    if ts is not None:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"), "sidecar"
    for dt, source in ((exif_datetime(raw), "exif"), (filename_datetime(path), "filename")):
        if dt:
            return dt.strftime("%Y-%m-%dT%H:%M:%S"), source
    return None, "none"


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


def measure_face(pts: np.ndarray, blendshapes, rgb: np.ndarray, bgr: np.ndarray,
                 recognizer, scale_to_src: float) -> dict:
    eye_a, eye_b = eye_centres(pts)
    axis = eye_b - eye_a
    eye_dist = float(np.hypot(*axis))
    unit = axis / eye_dist
    roll = math.degrees(math.atan2(axis[1], axis[0]))

    # Yaw proxy: project nose and both cheek edges onto the eye axis. A
    # frontal face has the nose midway between the cheeks; turning the head
    # slides it toward one side. Convention-free, unlike decomposing
    # MediaPipe's transform matrix.
    nose, ca, cb = (float(pts[i] @ unit) for i in (NOSE_TIP, CHEEK_A, CHEEK_B))
    da, db = abs(nose - ca), abs(cb - nose)
    yaw = (da - db) / (da + db) if da + db else 1.0

    blink = 0.0
    if blendshapes:
        scores = {c.category_name: c.score for c in blendshapes}
        blink = max(scores.get("eyeBlinkLeft", 0.0), scores.get("eyeBlinkRight", 0.0))

    # Sharpness: Laplacian variance of the face, resampled so the eyes are a
    # fixed 64px apart. Normalising scale makes blur comparable across a
    # close-up and a mid-shot; it also means a low-res face scores as soft,
    # which is the right answer for how it'll look once upscaled.
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    h, w = rgb.shape[:2]
    crop = rgb[max(0, int(y0)):min(h, int(y1) + 1), max(0, int(x0)):min(w, int(x1) + 1)]
    sharpness = 0.0
    if crop.size:
        f = 64 / max(eye_dist, 1)
        crop = cv2.resize(crop, (max(8, round(crop.shape[1] * f)), max(8, round(crop.shape[0] * f))),
                          interpolation=cv2.INTER_AREA)
        sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())

    return {
        "eye_a": [round(v * scale_to_src, 2) for v in eye_a],
        "eye_b": [round(v * scale_to_src, 2) for v in eye_b],
        "eye_dist": round(eye_dist * scale_to_src, 2),
        "roll": round(roll, 2),
        "yaw": round(yaw, 3),
        "blink": round(blink, 3),
        "sharpness": round(sharpness, 1),
        "embedding": embed(recognizer, bgr, pts),
    }


def scan_one(path: Path, landmarker, recognizer, sidecars: SidecarIndex) -> dict:
    raw, upright = load_image(path)
    taken, date_source = date_photo(path, raw, sidecars)
    src_w, src_h = upright.size
    record = {"taken": taken, "date_source": date_source, "width": src_w, "height": src_h}

    f = min(1.0, DETECT_MAX_SIDE / max(src_w, src_h))
    small = upright if f == 1 else upright.resize((round(src_w * f), round(src_h * f)), Image.LANCZOS)
    rgb = np.asarray(small)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    result = detect(landmarker, rgb)

    faces = [
        measure_face(face_points(lm, rgb.shape[1], rgb.shape[0]),
                     result.face_blendshapes[i] if result.face_blendshapes else None,
                     rgb, bgr, recognizer, 1 / f)
        for i, lm in enumerate(result.face_landmarks)
    ]
    faces.sort(key=lambda face: face["eye_dist"], reverse=True)
    record["faces"] = faces
    return record


def iter_images(sources: list[Path]):
    for source in sources:
        for path in sorted(source.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTS and path.is_file() and not path.name.startswith("."):
                yield path


def cmd_scan(args) -> None:
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    cache_path = work / "scan.jsonl"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["path"]] = rec

    sources = [Path(s).expanduser().resolve() for s in args.sources]
    for s in sources:
        if not s.is_dir():
            sys.exit(f"Not a folder: {s}")
    paths = list(iter_images(sources))
    todo = []
    for p in paths:
        st = p.stat()
        hit = cache.get(str(p))
        if (not hit or hit.get("v") != SCAN_VERSION or hit.get("size") != st.st_size
                or hit.get("mtime") != int(st.st_mtime)):
            todo.append((p, st))
    log(f"{len(paths)} images found, {len(paths) - len(todo)} already scanned, {len(todo)} to scan")
    if not todo:
        return

    # Six faces is plenty to find you in a group shot; the cap only bounds cost.
    landmarker = load_landmarker(num_faces=6)
    recognizer = load_recognizer()
    sidecars = SidecarIndex()
    started = time.time()
    # Append-only: Ctrl-C loses at most the photo in flight, and a re-run
    # picks up where this left off. Later lines for the same path win.
    with cache_path.open("a") as out:
        for i, (p, st) in enumerate(todo, 1):
            rec = {"path": str(p), "v": SCAN_VERSION, "size": st.st_size, "mtime": int(st.st_mtime)}
            try:
                rec.update(scan_one(p, landmarker, recognizer, sidecars))
                rec["status"] = "ok"
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop a 5k-photo scan
                rec.update(status="error", error=f"{type(exc).__name__}: {exc}")
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if i % 25 == 0 or i == len(todo):
                rate = i / (time.time() - started)
                log(f"  {i}/{len(todo)}  {rate:.1f} photos/s  ~{(len(todo) - i) / rate / 60:.1f} min left")


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


def load_scan(work: Path) -> list[dict]:
    cache_path = work / "scan.jsonl"
    if not cache_path.exists():
        sys.exit(f"No {cache_path}; run `scan` first.")
    latest: dict[str, dict] = {}
    for line in cache_path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            latest[rec["path"]] = rec
    records = [r for r in latest.values() if Path(r["path"]).exists()]
    stale = sum(1 for r in records if r.get("v") != SCAN_VERSION)
    if stale:
        sys.exit(f"{stale} photos were scanned by an older version; re-run `scan` on the same folders.")
    return records


def read_list(path: Path) -> set[str]:
    """exclude.txt / pin.txt: one path per line. A bare file name also
    matches, so lines can be pasted from review.html's labels.

    Only whole-line `#` comments: Photo Booth names contain ` #2`, so a
    trailing-comment rule would truncate them.
    """
    items = set()
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                items.add(line)
    return items


def read_dates(path: Path) -> dict[str, str]:
    """dates.txt: `<file name or path>  YYYY[-MM[-DD]]`, one per line.

    The date is the last whitespace-separated token so file names may
    contain spaces. A year alone lands mid-year and a month alone mid-month,
    so a rough guess sorts roughly where it belongs.
    """
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        m = re.fullmatch(r"(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?", parts[-1]) if len(parts) == 2 else None
        if not m:
            log(f"  dates.txt line {n} ignored (want `<name>  YYYY[-MM[-DD]]`): {line}")
            continue
        y, mo, d = int(m.group(1)), int(m.group(2) or 7), int(m.group(3) or (15 if m.group(2) else 1))
        out[parts[0].strip()] = datetime(y, mo, d, 12).strftime("%Y-%m-%dT%H:%M:%S")
    return out


def listed(rec: dict, items) -> bool:
    return rec["path"] in items or Path(rec["path"]).name in items


def period_key(taken: str, period: str) -> str:
    dt = datetime.fromisoformat(taken)
    if period == "week":
        y, w, _ = dt.isocalendar()
        return f"{y}-W{w:02d}"
    if period == "quarter":
        return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
    if period == "year":
        return f"{dt.year}"
    return f"{dt.year}-{dt.month:02d}"


def reference_embeddings(work: Path) -> np.ndarray:
    """Identity seed: every face-bearing photo in work/me/ (largest face)."""
    me = work / "me"
    paths = sorted(p for p in me.glob("*") if p.suffix.lower() in IMAGE_EXTS) if me.is_dir() else []
    if not paths:
        sys.exit(f"Put one or more clear photos of just you in {me}/ (the identity seed).")
    landmarker, recognizer = load_landmarker(num_faces=1), load_recognizer()
    refs = []
    for p in paths:
        _, upright = load_image(p)
        f = min(1.0, DETECT_MAX_SIDE / max(upright.size))
        if f < 1:
            upright = upright.resize((round(upright.width * f), round(upright.height * f)), Image.LANCZOS)
        rgb = np.asarray(upright)
        result = detect(landmarker, rgb)
        if not result.face_landmarks:
            log(f"  no face found in reference {p.name}, skipped")
            continue
        pts = face_points(result.face_landmarks[0], rgb.shape[1], rgb.shape[0])
        refs.append(embed(recognizer, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), pts))
    if not refs:
        sys.exit(f"No usable face in {me}/.")
    return np.array(refs, dtype=np.float32)


def assign_identity(records: list[dict], refs: np.ndarray) -> None:
    """Decide which face in each photo is you. Sets rec["me"] (face index or
    None), rec["me_best"] (best-matching index, for pins) and rec["me_sim"].

    Grows a set of known-you faces from the seed: each round, a photo's best
    face joins if it clears JOIN_MIN against the set and clearly beats every
    other face in that photo. Repeat until nothing new joins. Chaining this
    way recognises childhood photos a single adult reference never would,
    while the margin rule stops a friend who's always beside you from
    joining via a shared photo. Remaining photos are then matched once
    against the final set at the looser MATCH_MIN.
    """
    owners, vecs = [], []
    for ri, rec in enumerate(records):
        for fi, face in enumerate(rec.get("faces") or []):
            owners.append((ri, fi))
            vecs.append(face["embedding"])
    for rec in records:
        rec["me"], rec["me_best"], rec["me_sim"] = None, None, 0.0
    if not vecs:
        return
    emb = np.array(vecs, dtype=np.float32)
    faces_of: dict[int, list[int]] = defaultdict(list)
    for k, (ri, _) in enumerate(owners):
        faces_of[ri].append(k)

    members: list[int] = []
    joined: set[int] = set()

    def best_per_photo(threshold: float) -> dict[int, tuple[int, float]]:
        pool = np.vstack([refs, emb[members]]) if members else refs
        sims = emb @ pool.T
        if members:
            # A member must not match itself.
            sims[members, len(refs) + np.arange(len(members))] = -1
        best = sims.max(axis=1)
        picks = {}
        for ri, ks in faces_of.items():
            if ri in joined:
                continue
            order = sorted(ks, key=lambda k: best[k], reverse=True)
            top = order[0]
            runner = best[order[1]] if len(order) > 1 else -1.0
            records[ri]["me_best"], records[ri]["me_sim"] = owners[top][1], round(float(best[top]), 3)
            if best[top] >= threshold and best[top] - runner >= MATCH_MARGIN:
                picks[ri] = (top, float(best[top]))
        return picks

    for _ in range(50):
        picks = best_per_photo(JOIN_MIN)
        if not picks:
            break
        for ri, (k, _) in picks.items():
            joined.add(ri)
            members.append(k)
            records[ri]["me"] = owners[k][1]
    for ri, (k, _) in best_per_photo(MATCH_MIN).items():
        records[ri]["me"] = owners[k][1]
    for ri in joined:
        # me_best/me_sim were last written before the photo joined; they're
        # still its score at the time it was accepted, which is what the
        # review page should show.
        records[ri]["me_best"] = records[ri]["me"]


def evaluate(rec: dict, framing: Framing) -> tuple[str | None, float, dict]:
    """(reject reason or None, score in 0..1, your face in this photo)."""
    if rec.get("status") != "ok":
        return "unreadable", 0.0, {}
    faces = rec.get("faces") or []
    if not faces:
        return "no face", 0.0, {}
    face = faces[rec["me"] if rec["me"] is not None else rec["me_best"]]
    if not rec.get("taken"):
        return "no date", 0.0, face
    if rec["me"] is None:
        return "not recognised as you", 0.0, face
    m = framing.matrix(face["eye_a"], face["eye_b"])
    for other in faces:
        if other is face or other["eye_dist"] < OTHER_FACE_RATIO * face["eye_dist"]:
            continue
        centre = (np.asarray(other["eye_a"]) + np.asarray(other["eye_b"])) / 2
        if framing.in_frame(m, centre, OTHER_FACE_MARGIN):
            return "someone else in frame", 0.0, face
    if face["eye_dist"] < MIN_EYE_DIST_PX:
        return "face too small", 0.0, face
    if abs(face["yaw"]) > MAX_YAW:
        return "head turned", 0.0, face
    if abs(face["roll"]) > MAX_ROLL_DEG:
        return "head tilted", 0.0, face
    if face["blink"] > MAX_BLINK:
        return "eyes closed", 0.0, face
    coverage = framing.coverage(face["eye_a"], face["eye_b"], rec["width"], rec["height"])
    face = {**face, "coverage": round(coverage, 3)}
    if coverage < MIN_COVERAGE:
        return "face at photo edge", 0.0, face

    frontal = 1 - abs(face["yaw"]) / MAX_YAW
    # Resolution: 1.0 once the source eyes are at least as far apart as the
    # canvas slot, i.e. the frame never needs upscaling.
    resolution = min(1.0, face["eye_dist"] / framing.eye_px)
    sharp = min(1.0, max(0.0, (math.log10(max(face["sharpness"], 1)) - 1) / 2))
    eyes_open = 1 - face["blink"] / MAX_BLINK
    score = (0.35 * frontal + 0.25 * sharp + 0.25 * resolution + 0.15 * eyes_open) * coverage**3
    return None, round(score, 4), face


def aligned_thumb(rec: dict, face: dict, framing: Framing, thumbs: Path, size: int = 180) -> str:
    """Aligned preview, cached by path+mtime+framing, so review.html shows
    what each pick will actually look like as a frame."""
    key = hashlib.sha1(f"{rec['path']}|{rec['mtime']}|{framing.__dict__}|{face['eye_a']}".encode()).hexdigest()[:16]
    out = thumbs / f"{key}.jpg"
    if not out.exists():
        _, upright = load_image(Path(rec["path"]))
        img = framing.warp(np.asarray(upright), face["eye_a"], face["eye_b"], scale=size / framing.width)
        Image.fromarray(img).save(out, quality=82)
    return f"thumbs/{out.name}"


def plain_thumb(rec: dict, thumbs: Path, size: int = 140) -> str:
    """Whole-photo preview for rejects: the problem (a friend in frame, a
    turned head, the wrong face picked) is usually visible only uncropped."""
    key = hashlib.sha1(f"plain|{rec['path']}|{rec['mtime']}".encode()).hexdigest()[:16]
    out = thumbs / f"{key}.jpg"
    if not out.exists():
        _, upright = load_image(Path(rec["path"]))
        upright.thumbnail((size, size))
        upright.save(out, quality=78)
    return f"thumbs/{out.name}"


def cmd_select(args) -> None:
    work = Path(args.work)
    framing = framing_from(args)
    records = load_scan(work)
    exclude = read_list(work / "exclude.txt")
    pins = read_list(work / "pin.txt")
    dates = read_dates(work / "dates.txt")

    reasons: dict[str, int] = defaultdict(int)
    reasons["excluded"] = sum(1 for r in records if listed(r, exclude))
    # Excluded photos also stay out of the identity chain: excluding a
    # wrongly-matched photo is how you cut a bad link.
    records = [r for r in records if not listed(r, exclude)]
    for rec in records:
        override = dates.get(rec["path"]) or dates.get(Path(rec["path"]).name)
        if override:
            rec["taken"], rec["date_source"] = override, "dates.txt"

    assign_identity(records, reference_embeddings(work))

    # Keyed by taken time to the minute: Photo Booth only records minutes,
    # and its bursts (`#2`, `#3`) and duplicate copies (`(1)`) share one. The
    # best-scoring shot of each minute survives.
    usable: dict[str, tuple[float, dict, dict]] = {}
    rejected: list[tuple[dict, str, dict]] = []
    for rec in records:
        reason, score, face = evaluate(rec, framing)
        pinned = listed(rec, pins)
        if reason and pinned and face and rec.get("taken"):
            # A pin overrides the filters (you know better than a yaw proxy),
            # but still needs a face to align on and a date to sort by.
            reason = None
        if reason:
            reasons[reason] += 1
            rejected.append((rec, reason, face))
            continue
        key = rec["taken"][:16]
        prev = usable.get(key)
        if prev and (prev[0] >= score or listed(prev[1], pins)) and not pinned:
            reasons["duplicate"] += 1
            rejected.append((rec, "duplicate", face))
            continue
        if prev:
            reasons["duplicate"] += 1
            rejected.append((prev[1], "duplicate", prev[2]))
        usable[key] = (score, rec, face)

    by_period: dict[str, list[tuple[float, dict, dict]]] = defaultdict(list)
    for score, rec, face in usable.values():
        by_period[period_key(rec["taken"], "month" if args.period == "all" else args.period)].append(
            (score, rec, face))
    chosen: list[tuple[float, dict, dict]] = []
    for key, items in by_period.items():
        # Pins first, then by score, so a period's pick is at the front.
        items.sort(key=lambda t: (not listed(t[1], pins), -t[0]))
        if args.period == "all":
            chosen.extend(t for t in items if t[0] >= args.min_score or listed(t[1], pins))
        else:
            pinned = [t for t in items if listed(t[1], pins)]
            if pinned:
                chosen.extend(pinned)
            elif items[0][0] >= args.min_score:
                chosen.append(items[0])
    chosen.sort(key=lambda t: t[1]["taken"])

    with (work / "selection.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["taken", "score", "pinned", "path", "eye_a_x", "eye_a_y", "eye_b_x", "eye_b_y"])
        for score, rec, face in chosen:
            w.writerow([rec["taken"], score, int(listed(rec, pins)), rec["path"], *face["eye_a"], *face["eye_b"]])

    thumbs = work / "thumbs"
    thumbs.mkdir(exist_ok=True)
    write_review(work, chosen, by_period, rejected, reasons, framing, thumbs, pins, args)

    log(f"{len(records) + reasons['excluded']} scanned photos → {len(chosen)} frames")
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        if n:
            log(f"  {n:6d}  {reason}")
    if chosen:
        log(f"Span: {chosen[0][1]['taken'][:10]} → {chosen[-1][1]['taken'][:10]}")
    log(f"Review: {work / 'review.html'}")


def write_review(work, chosen, by_period, rejected, reasons, framing, thumbs, pins, args) -> None:
    """Static contact sheet, one section per period: usable photos (the
    frames, outlined green) cropped as they'll appear, then that period's
    rejects uncropped with their reason, and undated photos at the end.
    Pin/exclude buttons collect names into boxes at the top to paste into
    pin.txt / exclude.txt (or dates.txt for undated ones), then re-run
    `select`."""
    esc = html.escape
    chosen_paths = {c[1]["path"] for c in chosen}
    group_period = "month" if args.period == "all" else args.period
    rejected_by: dict[str, list] = defaultdict(list)
    for rec, reason, face in rejected:
        key = period_key(rec["taken"], group_period) if rec.get("taken") else "undated"
        rejected_by[key].append((rec, reason, face))

    def buttons(rec):
        name = esc(Path(rec["path"]).name)
        return (f'<button data-list="pin" data-name="{name}">pin</button>'
                f'<button data-list="exclude" data-name="{name}">exclude</button>')

    sections = []
    keys = sorted(set(by_period) | {k for k in rejected_by if k != "undated"})
    if "undated" in rejected_by:
        keys.append("undated")
    for key in keys:
        cells = []
        items = by_period.get(key, [])
        if args.period != "all":
            items = items[: args.alternates + 1]
        for score, rec, face in items:
            picked = rec["path"] in chosen_paths
            cells.append(
                f'<figure class="{"pick" if picked else ""}">'
                f'<img loading="lazy" src="{esc(aligned_thumb(rec, face, framing, thumbs))}" '
                f'title="{esc(rec["path"])}"><figcaption>{esc(rec["taken"][:16].replace("T", " "))} · '
                f'{score:.2f} · you {rec["me_sim"]:.2f}{" · pinned" if listed(rec, pins) else ""}<br>'
                f'<span class="name">{esc(Path(rec["path"]).name)}</span><br>{buttons(rec)}'
                f'</figcaption></figure>')
        rej = []
        for rec, reason, _ in sorted(rejected_by.get(key, []), key=lambda t: t[1]):
            match = f' · you {rec["me_sim"]:.2f}' if rec.get("faces") else ""
            rej.append(
                f'<figure class="rej"><img loading="lazy" src="{esc(plain_thumb(rec, thumbs))}" '
                f'title="{esc(rec["path"])}"><figcaption><b>{esc(reason)}</b>{match}<br>'
                f'<span class="name">{esc(Path(rec["path"]).name)}</span><br>{buttons(rec)}'
                f'</figcaption></figure>')
        n_rej = len(rejected_by.get(key, []))
        heading = "undated: add these to dates.txt" if key == "undated" else key
        sections.append(
            f'<section><h2>{esc(heading)} <small>{len(by_period.get(key, []))} usable · {n_rej} rejected</small></h2>'
            f'<div class="row">{"".join(cells)}</div>'
            + (f'<details{" open" if key == "undated" else ""}><summary>{n_rej} rejected</summary>'
               f'<div class="row">{"".join(rej)}</div></details>' if rej else "")
            + '</section>')

    summary = " · ".join(f"{esc(k)}: {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]) if v)
    mode = "every usable photo" if args.period == "all" else f"best per {esc(args.period)}"
    page = f"""<!doctype html><meta charset="utf-8"><title>Facelapse review</title>
<style>
 body{{font:13px system-ui;margin:16px;background:#111;color:#ddd}}
 h1{{font-size:18px}} h2{{font-size:14px;margin:22px 0 6px}} small{{color:#888;font-weight:400}}
 .row{{display:flex;gap:8px;flex-wrap:wrap}}
 figure{{margin:0;width:180px;opacity:.7}} figure.pick{{opacity:1;outline:3px solid #4c9}}
 figure.rej{{width:140px}} figure.rej img{{width:140px}}
 img{{width:180px;height:auto;display:block}} figcaption{{font-size:11px;padding:3px 0}}
 .name{{color:#999;word-break:break-all}} button{{font-size:11px;margin:2px 2px 0 0}}
 details{{margin-top:6px}} summary{{cursor:pointer;color:#aaa}}
 textarea{{width:100%;height:70px;background:#222;color:#ddd;font:11px ui-monospace,monospace}}
 .lists{{display:grid;grid-template-columns:1fr 1fr;gap:12px;position:sticky;top:0;background:#111;padding:6px 0;z-index:1}}
</style>
<h1>Facelapse review: {len(chosen)} frames ({mode})</h1>
<p>Green outline = a frame. "you" = identity match (≥{MATCH_MIN} counts). Rejected: {summary}</p>
<div class="lists">
 <label>append to pin.txt<textarea id="pin"></textarea></label>
 <label>append to exclude.txt<textarea id="exclude"></textarea></label>
</div>
{"".join(sections)}
<script>
document.addEventListener('click', e => {{
  const b = e.target.closest('button[data-list]'); if (!b) return;
  const box = document.getElementById(b.dataset.list);
  if (!box.value.split('\\n').includes(b.dataset.name)) box.value += b.dataset.name + '\\n';
  b.disabled = true;
}});
</script>"""
    (work / "review.html").write_text(page, encoding="utf-8")


# ---------------------------------------------------------------------------
# align
# ---------------------------------------------------------------------------


def refine_eyes(landmarker, upright: np.ndarray, eye_a, eye_b):
    """Re-detect on a full-resolution crop around the known face.

    scan detected on a ≤1600px copy, so its eye points carry up to a couple of
    source pixels of quantisation, which shows up as frame-to-frame jitter
    once upscaled. A face-sized crop at 768px puts far more pixels on the
    eyes. Falls back to the scan points if the crop somehow finds no face.
    """
    eye_a, eye_b = np.asarray(eye_a, float), np.asarray(eye_b, float)
    centre = (eye_a + eye_b) / 2
    half = 2.2 * np.hypot(*(eye_b - eye_a))
    h, w = upright.shape[:2]
    x0, y0 = int(max(0, centre[0] - half)), int(max(0, centre[1] - half))
    x1, y1 = int(min(w, centre[0] + half)), int(min(h, centre[1] + half))
    crop = upright[y0:y1, x0:x1]
    f = REFINE_CROP_SIDE / max(crop.shape[:2])
    crop = cv2.resize(crop, (round(crop.shape[1] * f), round(crop.shape[0] * f)),
                      interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    result = detect(landmarker, crop)
    if not result.face_landmarks:
        return eye_a, eye_b, False
    # The crop can still contain a neighbour's face; take the one nearest the
    # face select identified as you.
    best = None
    for lm in result.face_landmarks:
        a, b = eye_centres(face_points(lm, crop.shape[1], crop.shape[0]))
        a, b = a / f + (x0, y0), b / f + (x0, y0)
        d = np.hypot(*((a + b) / 2 - centre))
        if best is None or d < best[0]:
            best = (d, a, b)
    return best[1], best[2], True


def cmd_align(args) -> None:
    work = Path(args.work)
    framing = framing_from(args)
    sel_path = work / "selection.csv"
    if not sel_path.exists():
        sys.exit(f"No {sel_path}; run `select` first.")
    rows = list(csv.DictReader(sel_path.open()))
    frames = work / "frames"
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir()

    landmarker = load_landmarker(num_faces=3)
    with (work / "frames.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "taken", "path"])
        for i, row in enumerate(rows, 1):
            _, upright = load_image(Path(row["path"]))
            rgb = np.asarray(upright)
            eye_a = (float(row["eye_a_x"]), float(row["eye_a_y"]))
            eye_b = (float(row["eye_b_x"]), float(row["eye_b_y"]))
            if not args.no_refine:
                eye_a, eye_b, ok = refine_eyes(landmarker, rgb, eye_a, eye_b)
                if not ok:
                    log(f"  refine found no face, using scan points: {row['path']}")
            out = frames / f"{i:05d}.jpg"
            Image.fromarray(framing.warp(rgb, eye_a, eye_b)).save(out, quality=93)
            w.writerow([out.name, row["taken"], row["path"]])
            if i % 25 == 0 or i == len(rows):
                log(f"  aligned {i}/{len(rows)}")
    if args.color > 0:
        balance_colour(sorted(frames.glob("*.jpg")), framing, args.color)
    log(f"Frames: {frames}")


def face_mask(framing: Framing) -> np.ndarray:
    """Ellipse covering brows to chin. Every frame is aligned, so the face is
    in the same place in all of them; measuring only there keeps a bright sky
    or a dark bar behind you from driving the correction."""
    mask = np.zeros((framing.height, framing.width), np.uint8)
    cx = round(framing.width / 2)
    cy = round(framing.eye_a[1] + 0.5 * framing.eye_px)
    cv2.ellipse(mask, (cx, cy), (round(0.9 * framing.eye_px), round(1.3 * framing.eye_px)), 0, 0, 360, 255, -1)
    return mask > 0


def balance_colour(paths: list[Path], framing: Framing, strength: float) -> None:
    """Pull each frame's face brightness, contrast and colour cast toward the
    median across all frames, in Lab space.

    Brightness (L) is corrected with a gamma curve chosen to move the face's
    mean to the target, not a linear gain: gamma fixes 0 and 100, so on a
    backlit shot the face comes up without blowing the sky behind it to
    white (a linear gain did exactly that on the first render). Colour (a, b) gets a
    mean shift only: it removes warm/cool/green casts from different cameras
    and lighting without flattening real skin-tone differences (a summer tan
    is part of the story). `strength` blends between untouched (0) and fully
    matched (1); a little left over keeps it from looking processed.
    """
    mask = face_mask(framing)
    stats = []
    for p in paths:
        lab = cv2.cvtColor(np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        face = lab[mask]
        stats.append((face.mean(axis=0), face.std(axis=0)))
    means = np.array([m for m, _ in stats])
    target_mean = np.median(means, axis=0)

    for p, (mean, _std) in zip(paths, stats):
        rgb = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        # L' = 100 * (L/100)^g maps the face mean onto the target. Clamped:
        # a near-black face can't be rescued without dragging up noise.
        m, t = np.clip([mean[0], target_mean[0]], 1, 99) / 100
        gamma = float(np.clip(math.log(t) / math.log(m), 0.5, 2.0))
        fixed = lab.copy()
        fixed[..., 0] = 100 * np.power(np.clip(lab[..., 0], 0, 100) / 100, gamma)
        fixed[..., 1:] = lab[..., 1:] - mean[1:] + target_mean[1:]
        lab = lab + strength * (fixed - lab)
        lab[..., 0] = np.clip(lab[..., 0], 0, 100)
        out = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1)
        Image.fromarray((out * 255 + 0.5).astype(np.uint8)).save(p, quality=93)
    log(f"  colour balanced {len(paths)} frames toward the median face (strength {strength})")


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------


def cmd_encode(args) -> None:
    work = Path(args.work)
    frames = work / "frames"
    first = frames / "00001.jpg"
    if not first.exists():
        sys.exit(f"No frames in {frames}; run `align` first.")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found; `brew install ffmpeg`.")
    last = sorted(frames.glob("*.jpg"))[-1]

    # `framerate` blends neighbouring frames while resampling to 30fps, which
    # gives a crossfade; scene=100 stops it treating every cut between two
    # different photos as a scene change (they all are) and skipping the
    # blend. Then hold the final frame so the video settles on today's face,
    # which is also what the poster shows.
    filters = []
    if args.blend:
        filters.append("framerate=fps=30:interp_start=0:interp_end=255:scene=100")
    filters.append(f"tpad=stop_mode=clone:stop_duration={args.hold}")
    filters.append("format=yuv420p")
    vf = ",".join(filters)
    inputs = ["-framerate", str(args.fps), "-i", str(frames / "%05d.jpg")]

    mp4 = work / "facelapse.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-vf", vf,
         "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
         "-movflags", "+faststart", "-an", str(mp4)],
        check=True,
    )
    outputs = [mp4]
    if args.webm:
        webm = work / "facelapse.webm"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-vf", vf,
             "-c:v", "libvpx-vp9", "-crf", str(args.crf + 10), "-b:v", "0",
             "-row-mt", "1", "-an", str(webm)],
            check=True,
        )
        outputs.append(webm)

    poster = Image.open(last)
    poster.save(work / "poster.jpg", quality=88)
    poster.save(work / "poster.webp", quality=82)
    outputs += [work / "poster.jpg", work / "poster.webp"]

    n = len(list(frames.glob("*.jpg")))
    log(f"{n} frames at {args.fps}/s = {n / args.fps + args.hold:.1f}s")
    for p in outputs:
        log(f"  {p}  {p.stat().st_size / 1e6:.2f} MB")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", default=str(ROOT / "work"), help="working/output folder (default: ./work)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan", help="detect, fingerprint + date every photo (cached, resumable)")
    p.add_argument("sources", nargs="+", help="photo folders")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("select", help="find you, filter, pick frames, write review.html")
    add_framing_args(p)
    # "all" by default (decided on #457): every usable photo becomes a frame
    # in date order, so dense stretches play longer than sparse ones. The
    # per-period modes remain for a time-even cut.
    p.add_argument("--period", choices=["all", "week", "month", "quarter", "year"], default="all")
    p.add_argument("--min-score", type=float, default=0.0,
                   help="drop frames scoring below this (per-period modes: leave the period empty)")
    p.add_argument("--alternates", type=int, default=4,
                   help="per-period modes: runners-up shown per period in review.html")
    p.set_defaults(fn=cmd_select)

    p = sub.add_parser("align", help="warp selected photos into eye-aligned frames")
    add_framing_args(p)
    p.add_argument("--no-refine", action="store_true", help="skip the full-res eye re-detection")
    p.add_argument("--color", type=float, default=0.8,
                   help="pull face brightness/colour toward the median frame (0 = off, 1 = full)")
    p.set_defaults(fn=cmd_align)

    p = sub.add_parser("encode", help="frames → MP4 (+ optional WebM) and poster")
    # 12/s, up from 8 after the first real render (#457) read as too slow.
    p.add_argument("--fps", type=float, default=12, help="photos per second")
    p.add_argument("--hold", type=float, default=1.5, help="seconds to hold the final frame")
    p.add_argument("--blend", action="store_true", help="crossfade between photos")
    p.add_argument("--crf", type=int, default=24, help="x264 quality (lower = bigger, sharper)")
    p.add_argument("--webm", action="store_true", help="also write a VP9 WebM")
    p.set_defaults(fn=cmd_encode)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
