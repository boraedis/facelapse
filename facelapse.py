#!/usr/bin/env python3
"""Facelapse: turn a pile of photos of one person into an eye-aligned timelapse.

Built for data-diary#457 (the landing-page facelapse, epic #15). Deliberately
a standalone offline tool, not part of the data-diary app: it runs maybe once
a year, on a laptop, against a local Google Photos export.

Pipeline (each step reads the previous step's output from --work):

  scan    Walk one or more photo folders, date every photo, detect faces with
          MediaPipe, and cache per-photo measurements in scan.jsonl. Resumable
          and incremental: a re-run only processes new or changed files.
  select  Filter out unusable shots (group photos, profiles, closed eyes, tiny
          faces), score the rest, and pick the best photo per period. Writes
          selection.csv plus review.html, a contact sheet for checking picks.
  align   Re-detect eyes precisely on each selected photo and warp it so the
          eyes land on fixed canvas coordinates. Writes frames/ + frames.csv.
  encode  ffmpeg the frames into an MP4 (optionally WebM) plus a poster image
          of the final frame.

Manual control lives in two plain-text files in --work, one path per line,
both honoured by `select` on every run so choices survive a yearly re-run:

  exclude.txt   never use these photos
  pin.txt       always use these photos (a pinned photo wins its period; pin
                several in one period to keep all of them, e.g. sparse years)

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
from datetime import datetime, timezone
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
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}

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
CHEEK_A = 234
CHEEK_B = 454

# --- Filter thresholds (select). Tune here after looking at review.html. ---
# A face counts toward "group photo" only if it's at least this fraction of
# the largest face's size, so strangers in the background don't reject a shot.
GROUP_FACE_RATIO = 0.45
# Minimum eye-centre distance in source pixels. Below this the face has to be
# upscaled so far into the canvas that it reads as mush.
MIN_EYE_DIST_PX = 70
# Head-turn proxy in [-1, 1]: nose-to-cheek asymmetry along the eye line.
# ~0 is frontal; 0.2 is a noticeable three-quarter turn.
MAX_YAW = 0.22
# Head tilt in degrees. Alignment removes roll entirely, but a heavily tilted
# head usually means a lying-down or goofy shot that looks off once levelled.
MAX_ROLL_DEG = 25
# MediaPipe blendshape eyeBlink score (0 open .. 1 closed). Squints score
# ~0.3-0.4, so this rejects only genuinely closed eyes.
MAX_BLINK = 0.5
# Fraction of the output canvas the source photo must cover once aligned.
# Faces near a photo's edge leave a gap that gets filled with smeared edge
# pixels; a sliver is fine, a quarter of the frame is not.
MIN_COVERAGE = 0.9


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_landmarker(num_faces: int) -> vision.FaceLandmarker:
    if not MODEL_PATH.exists():
        sys.exit(
            f"Missing model: {MODEL_PATH}\n"
            "Download it with:\n  curl -L -o models/face_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task"
        )
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


def load_image(path: Path) -> tuple[Image.Image, Image.Image]:
    """Return (raw, upright RGB). The raw image keeps its EXIF for dating.

    exif_transpose matters: phone photos are usually stored sideways with an
    orientation tag, and every coordinate this tool records is in upright
    space, so detection and warping must both see the same upright pixels.
    """
    raw = Image.open(path)
    upright = ImageOps.exif_transpose(raw).convert("RGB")
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
    never used: after a Takeout download it's the download time.
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


def measure_face(pts: np.ndarray, blendshapes, rgb: np.ndarray, scale_to_src: float) -> dict:
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
    }


def scan_one(path: Path, landmarker, sidecars: SidecarIndex) -> dict:
    raw, upright = load_image(path)
    taken, date_source = date_photo(path, raw, sidecars)
    src_w, src_h = upright.size
    record = {"taken": taken, "date_source": date_source, "width": src_w, "height": src_h}

    f = min(1.0, DETECT_MAX_SIDE / max(src_w, src_h))
    small = upright if f == 1 else upright.resize((round(src_w * f), round(src_h * f)), Image.LANCZOS)
    rgb = np.asarray(small)
    result = detect(landmarker, rgb)

    faces = [
        measure_face(face_points(lm, rgb.shape[1], rgb.shape[0]),
                     result.face_blendshapes[i] if result.face_blendshapes else None,
                     rgb, 1 / f)
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
        if not hit or hit.get("size") != st.st_size or hit.get("mtime") != int(st.st_mtime):
            todo.append((p, st))
    log(f"{len(paths)} images found, {len(paths) - len(todo)} already scanned, {len(todo)} to scan")
    if not todo:
        return

    # Six faces is plenty to recognise a group shot; the cap only bounds cost.
    landmarker = load_landmarker(num_faces=6)
    sidecars = SidecarIndex()
    started = time.time()
    # Append-only: Ctrl-C loses at most the photo in flight, and a re-run
    # picks up where this left off. Later lines for the same path win.
    with cache_path.open("a") as out:
        for i, (p, st) in enumerate(todo, 1):
            rec = {"path": str(p), "size": st.st_size, "mtime": int(st.st_mtime)}
            try:
                rec.update(scan_one(p, landmarker, sidecars))
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
    return [r for r in latest.values() if Path(r["path"]).exists()]


def read_list(path: Path) -> set[str]:
    """exclude.txt / pin.txt: one path per line; `#` comments. A bare file
    name also matches, so lines can be pasted from review.html's labels."""
    if not path.exists():
        return set()
    items = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            items.add(line)
    return items


def listed(rec: dict, items: set[str]) -> bool:
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


def evaluate(rec: dict, framing: Framing) -> tuple[str | None, float, dict]:
    """(reject reason or None, score in 0..1, primary face)."""
    if rec.get("status") != "ok":
        return "unreadable", 0.0, {}
    if not rec.get("taken"):
        return "no date", 0.0, {}
    faces = rec.get("faces") or []
    if not faces:
        return "no face", 0.0, {}
    face = faces[0]
    significant = [f for f in faces if f["eye_dist"] >= GROUP_FACE_RATIO * face["eye_dist"]]
    if len(significant) > 1:
        return "group photo", 0.0, face
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


def thumb(rec: dict, face: dict, framing: Framing, thumbs: Path, size: int = 180) -> str:
    """Aligned preview thumbnail, cached by path+mtime. Previews the real
    framing so review.html shows what each pick will look like as a frame."""
    key = hashlib.sha1(f"{rec['path']}|{rec['mtime']}|{framing.__dict__}".encode()).hexdigest()[:16]
    out = thumbs / f"{key}.jpg"
    if not out.exists():
        _, upright = load_image(Path(rec["path"]))
        img = framing.warp(np.asarray(upright), face["eye_a"], face["eye_b"], scale=size / framing.width)
        Image.fromarray(img).save(out, quality=82)
    return f"thumbs/{out.name}"


def cmd_select(args) -> None:
    work = Path(args.work)
    framing = framing_from(args)
    records = load_scan(work)
    exclude = read_list(work / "exclude.txt")
    pins = read_list(work / "pin.txt")

    reasons: dict[str, int] = defaultdict(int)
    by_period: dict[str, list[tuple[float, dict, dict]]] = defaultdict(list)
    rejected_by_period: dict[str, int] = defaultdict(int)
    pinned_by_period: dict[str, list[tuple[float, dict, dict]]] = defaultdict(list)

    # Same timestamp to the second = the same shot (an edited copy, or the
    # photo sitting in two albums). Keep only the better-scoring copy.
    seen_taken: dict[str, tuple[float, dict, dict]] = {}
    for rec in records:
        if listed(rec, exclude):
            reasons["excluded"] += 1
            continue
        reason, score, face = evaluate(rec, framing)
        pinned = listed(rec, pins)
        if reason and not (pinned and face and rec.get("taken")):
            reasons[reason] += 1
            if rec.get("taken"):
                rejected_by_period[period_key(rec["taken"], args.period)] += 1
            continue
        if pinned and reason:
            # A pin overrides the filters (you know better than a yaw proxy),
            # but still needs a detected face to align on and a date to sort by.
            face = {**face, "coverage": face.get("coverage", 1.0)}
        prev = seen_taken.get(rec["taken"])
        if prev and prev[0] >= score:
            reasons["duplicate"] += 1
            continue
        if prev:
            reasons["duplicate"] += 1
        seen_taken[rec["taken"]] = (score, rec, face)

    for score, rec, face in seen_taken.values():
        key = period_key(rec["taken"], args.period)
        if listed(rec, pins):
            pinned_by_period[key].append((score, rec, face))
        else:
            by_period[key].append((score, rec, face))

    chosen: list[tuple[str, float, dict, dict]] = []
    periods = sorted(set(by_period) | set(pinned_by_period))
    for key in periods:
        by_period[key].sort(key=lambda t: t[0], reverse=True)
        if pinned_by_period[key]:
            picks = pinned_by_period[key]
        elif by_period[key] and by_period[key][0][0] >= args.min_score:
            picks = by_period[key][:1]
        else:
            picks = []
        chosen.extend((key, s, r, f) for s, r, f in picks)
    chosen.sort(key=lambda t: t[2]["taken"])

    with (work / "selection.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["period", "taken", "score", "pinned", "path", "eye_a_x", "eye_a_y", "eye_b_x", "eye_b_y"])
        for key, score, rec, face in chosen:
            w.writerow([key, rec["taken"], score, int(listed(rec, pins)), rec["path"],
                        *face["eye_a"], *face["eye_b"]])

    thumbs = work / "thumbs"
    thumbs.mkdir(exist_ok=True)
    write_review(work, periods, chosen, by_period, pinned_by_period, rejected_by_period,
                 reasons, framing, thumbs, args)

    total = len(records)
    log(f"{total} scanned photos → {len(seen_taken)} usable → {len(chosen)} frames "
        f"across {len({c[0] for c in chosen})} of {len(periods)} {args.period}s with candidates")
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        log(f"  {n:6d}  {reason}")
    if chosen:
        log(f"Span: {chosen[0][2]['taken'][:10]} → {chosen[-1][2]['taken'][:10]}")
    log(f"Review: {work / 'review.html'}")


def write_review(work, periods, chosen, by_period, pinned_by_period, rejected_by_period,
                 reasons, framing, thumbs, args) -> None:
    """Static contact sheet: one row per period, the pick first, then the
    runners-up. Pin/exclude buttons collect paths into two boxes at the top
    to paste into pin.txt / exclude.txt, then re-run `select`."""
    chosen_paths = {c[2]["path"] for c in chosen}
    esc = html.escape
    rows = []
    for key in periods:
        cands = pinned_by_period[key] + by_period[key][: args.alternates + 1]
        cells = []
        for score, rec, face in cands:
            src = thumb(rec, face, framing, thumbs)
            picked = rec["path"] in chosen_paths
            cells.append(
                f'<figure class="{"pick" if picked else ""}">'
                f'<img loading="lazy" src="{esc(src)}" title="{esc(rec["path"])}">'
                f'<figcaption>{esc(rec["taken"][:10])} · {score:.2f}<br>'
                f'<span class="name">{esc(Path(rec["path"]).name)}</span><br>'
                f'<button data-list="pin" data-path="{esc(rec["path"])}">pin</button>'
                f'<button data-list="exclude" data-path="{esc(rec["path"])}">exclude</button>'
                f'</figcaption></figure>'
            )
        n_cands = len(by_period[key]) + len(pinned_by_period[key])
        rows.append(
            f'<section><h2>{esc(key)} <small>{n_cands} usable · '
            f'{rejected_by_period.get(key, 0)} rejected</small></h2>'
            f'<div class="row">{"".join(cells) or "<em>nothing usable</em>"}</div></section>'
        )
    summary = " · ".join(f"{esc(k)}: {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]))
    page = f"""<!doctype html><meta charset="utf-8"><title>Facelapse review</title>
<style>
 body{{font:13px system-ui;margin:16px;background:#111;color:#ddd}}
 h1{{font-size:18px}} h2{{font-size:14px;margin:18px 0 6px}} small{{color:#888;font-weight:400}}
 .row{{display:flex;gap:8px;flex-wrap:wrap}}
 figure{{margin:0;width:180px;opacity:.75}} figure.pick{{opacity:1;outline:3px solid #4c9;}}
 img{{width:180px;height:auto;display:block}} figcaption{{font-size:11px;padding:3px 0}}
 .name{{color:#999;word-break:break-all}} button{{font-size:11px;margin:2px 2px 0 0}}
 textarea{{width:100%;height:70px;background:#222;color:#ddd;font:11px ui-monospace,monospace}}
 .lists{{display:grid;grid-template-columns:1fr 1fr;gap:12px;position:sticky;top:0;background:#111;padding:6px 0}}
</style>
<h1>Facelapse review — {len(chosen)} frames, {esc(args.period)}ly</h1>
<p>Green outline = current pick. Rejected: {summary}</p>
<div class="lists">
 <label>append to pin.txt<textarea id="pin"></textarea></label>
 <label>append to exclude.txt<textarea id="exclude"></textarea></label>
</div>
{"".join(rows)}
<script>
document.addEventListener('click', e => {{
  const b = e.target.closest('button[data-list]'); if (!b) return;
  const box = document.getElementById(b.dataset.list);
  if (!box.value.includes(b.dataset.path)) box.value += b.dataset.path + '\\n';
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
    # face we came for.
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
    log(f"Frames: {frames}")


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

    p = sub.add_parser("scan", help="detect + date every photo (cached, resumable)")
    p.add_argument("sources", nargs="+", help="photo folders, e.g. an unzipped Takeout album")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("select", help="filter, score, pick one per period, write review.html")
    add_framing_args(p)
    p.add_argument("--period", choices=["week", "month", "quarter", "year"], default="month")
    p.add_argument("--min-score", type=float, default=0.0,
                   help="leave a period empty rather than use a pick scoring below this")
    p.add_argument("--alternates", type=int, default=4, help="runners-up shown per period in review.html")
    p.set_defaults(fn=cmd_select)

    p = sub.add_parser("align", help="warp selected photos into eye-aligned frames")
    add_framing_args(p)
    p.add_argument("--no-refine", action="store_true", help="skip the full-res eye re-detection")
    p.set_defaults(fn=cmd_align)

    p = sub.add_parser("encode", help="frames → MP4 (+ optional WebM) and poster")
    p.add_argument("--fps", type=float, default=8, help="photos per second")
    p.add_argument("--hold", type=float, default=1.5, help="seconds to hold the final frame")
    p.add_argument("--blend", action="store_true", help="crossfade between photos")
    p.add_argument("--crf", type=int, default=24, help="x264 quality (lower = bigger, sharper)")
    p.add_argument("--webm", action="store_true", help="also write a VP9 WebM")
    p.set_defaults(fn=cmd_encode)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
