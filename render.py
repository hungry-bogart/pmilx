# ------------------------------------------------
# PYTHON AUDIO-REACTIVE POLYGON RENDERER
# Multi-layer, trail-enabled, fast and hackable
# ------------------------------------------------

import numpy as np
import librosa
import cv2
import math
import subprocess
from scipy.interpolate import interp1d
from tqdm import tqdm
import random

# ================================
# CONFIGURATION
# ================================
# File names and directory information
AUDIO_FILE = "./audio/example.mp3"
OUT_FILE = "./output/no-audio.mp4"
FINAL_FILE = "./output/final-video-audio.mp4"

# ================================
# PALLETTE
# ================================
# Royal Frost palette (BGR for OpenCV)
# Create your own pallette here

ROYAL_FROST = {
    "violet": (169, 81, 120),   # #7851A9
    "steel":  (180, 130, 70),   # #4682B4
    "cadet":  (160, 158, 95),   # #5F9EA0
    "corn":   (237, 149, 100),  # #6495ED
    "sky":    (235, 206, 135),  # #87CEEB
}

# =========================
# PALLETTE MUTATION
# =========================

PALETTES = [
    [ROYAL_FROST["violet"], ROYAL_FROST["steel"], ROYAL_FROST["sky"]],
    [ROYAL_FROST["cadet"], ROYAL_FROST["corn"], ROYAL_FROST["violet"]],
    [ROYAL_FROST["sky"], ROYAL_FROST["steel"], ROYAL_FROST["cadet"]],
    ]

# Video
WIDTH, HEIGHT = 1280, 720
FPS = 60

BASS_SCALE = 1.8
MID_ROT_SPEED = 0.25

# Camera / Hyperwave
ZOOM_SCALE = 0.25
DRIFT_SCALE = 120
DRIFT_SPEED_X = 0.7
DRIFT_SPEED_Y = 0.9

# Glow
GLOW_BLUR = 21          # odd number
GLOW_STRENGTH = 0.8     # base strength
GLOW_AUDIO_BOOST = 1.2  # how much bass pushes glow


# Polygon layers
layers = [
    {
        "sides": 8,
        "radius": 120,
        "interval": 0.5,
        "lifetime": 0.6,
        "rot_speed": 0.02,
        "band": "bass",
        "color": ROYAL_FROST["violet"],
    },
    {
        "sides": 10,
        "radius": 90,
        "interval": 0.7,
        "lifetime": 0.7,
        "rot_speed": 0.03,
        "band": "mid",
        "color": ROYAL_FROST["steel"],
    },
    {
        "sides": 12,
        "radius": 60,
        "interval": 0.9,
        "lifetime": 0.8,
        "rot_speed": 0.04,
        "band": "treb",
        "color": ROYAL_FROST["sky"],
    },
]

polygons_per_layer = [[] for _ in layers]
last_spawn_per_layer = [-999 for _ in layers]

# ===============================
# GLOBAL CHAOS STATE
# ===============================
CHAOS = {
    "palette": 0,
    "camera_mode": 0,
    "rotation_bias": 1.0,
    "glow_bias": 1.0,
}

CHAOS_LOCKED = {
    "palette": False,
    "camera_mode": False,
}


# ===============================
# AUDIO ANALYSIS
# ===============================
y, sr = librosa.load(AUDIO_FILE, mono=True, sr=44100)
hop_length = int(sr / FPS)
audio_duration = librosa.get_duration(y=y, sr=sr)
num_frames = int(np.ceil(audio_duration * FPS))

stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

bass_energy = stft[freqs < 120].mean(axis=0)
mid_energy = stft[(freqs >= 120) & (freqs < 2000)].mean(axis=0)
treb_energy = stft[freqs >= 2000].mean(axis=0)


def norm(x):
    return x / (x.max() + 1e-6)


bass_energy = norm(bass_energy)
mid_energy = norm(mid_energy)
treb_energy = norm(treb_energy)

orig_frames = np.arange(len(bass_energy))
target_frames = np.linspace(0, len(bass_energy) - 1, num_frames)

bass_energy = interp1d(orig_frames, bass_energy)(target_frames)
mid_energy = interp1d(orig_frames, mid_energy)(target_frames)
treb_energy = interp1d(orig_frames, treb_energy)(target_frames)

# ===============================
# VIDEO OUTPUT
# ===============================
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUT_FILE, fourcc, FPS, (WIDTH, HEIGHT))

# ===============================
# DRAWING HELPERS
# ===============================


def draw_polygon(img, cx, cy, radius, sides, rot, color, alpha):
    pts = []
    for i in range(sides):
        a = rot + 2 * math.pi * i / sides

        x = int(cx + math.cos(a) * radius)
        y = int(cy + math.sin(a) * radius)

        pts.append([x, y])

    pts = np.array([pts], np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, pts, color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def apply_camera(img, zoom=1.0, pan_x=0.0, pan_y=0.0, rot=0.0):
    h, w = img.shape[:2]

    # Center
    cx, cy = w // 2, h // 2

    # Build transform matrix
    M = cv2.getRotationMatrix2D((cx, cy), rot, zoom)

    # Apply pan in pixels
    M[0, 2] += pan_x
    M[1, 2] += pan_y

    # Warp safely, always same size
    out = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    return out

# ===============================
# GLOW
# ===============================


def apply_glow_float(img_f, strength, threshold=0.6):
    # Compute luminance
    luminance = img_f.max(axis=2, keepdims=True)  # (H, W, 1)

    # Only bright areas glow
    mask = (luminance > threshold).astype(np.float32)

    # Blur luminance only
    glow = cv2.GaussianBlur(
        (luminance * mask).squeeze(),  # 2D for OpenCV
        (GLOW_BLUR, GLOW_BLUR),
        0
    )

    # Expand glow back to 3 channels
    glow = np.repeat(glow[:, :, None], 3, axis=2)  # now (H, W, 3)

    # Sharper radial decay
    h, w = glow.shape[:2]
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    attenuation = np.clip((dist / max_dist) ** 1.5, 0.0, 1.0)

    # Broadcast to match glow channels
    glow *= attenuation[:, :, None]

    # Add glow equally to all channels
    out = img_f + glow * strength

    return np.clip(out, 0, 1)


# ===============================
# AUDIO TRIGGERS
# ===============================

def bass_drop(bass, threshold=0.85):
    return bass > threshold

# ===============================
# MAIN LOOP
# ===============================


# Create the trail buffer
trail = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)

# Start the loop
for frame in tqdm(range(num_frames)):
    t = frame / FPS

    # =========================
    # 1. Background
    # =========================
    bg = np.full(
        (HEIGHT, WIDTH, 3),
        (80, 76, 84),  # your chosen background
        dtype=np.uint8
    )

    # =========================
    # 2. Foreground (empty)
    # =========================
    fg = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    # =========================
    # 3. Audio values
    # =========================
    bass = bass_energy[frame]
    mid = mid_energy[frame]
    treb = treb_energy[frame]

    band = {
        "bass": bass,
        "mid": mid,
        "treb": treb
    }

    # =========================
    # 3b.CHAOS TRIGGERS
    # =========================

    if bass_drop(bass) and not CHAOS_LOCKED["palette"]:
        CHAOS["palette"] = random.randint(0, 2)
        CHAOS["glow_bias"] = random.uniform(0.8, 1.8)
        CHAOS_LOCKED["palette"] = True

    # =========================
    # 4. Geometry
    # =========================
    for i, layer in enumerate(layers):
        if t - last_spawn_per_layer[i] >= layer["interval"]:
            polygons_per_layer[i].append({"birth": t, "rot": 0.0})
            last_spawn_per_layer[i] = t

        alive = []
        for p in polygons_per_layer[i]:
            age = t - p["birth"]
            if age > layer["lifetime"]:
                continue

            fade = math.sin(math.pi * age / layer["lifetime"])
            radius = layer["radius"] * (1 + band[layer["band"]])

            p["rot"] += layer["rot_speed"]

            color = PALETTES[CHAOS["palette"]][i]

            draw_polygon(
                fg,
                WIDTH // 2,
                HEIGHT // 2,
                radius,
                layer["sides"],
                p["rot"],
                color,
                fade
            )

            alive.append(p)

        polygons_per_layer[i] = alive

    # =========================
    # 5. Camera (safe, pure)
    # =========================

    # Audio-driven camera motion
    zoom = 1.0 + bass * 0.15
    pan_x = math.sin(t * 0.6) * mid * 40
    pan_y = math.cos(t * 0.8) * treb * 30
    rot = math.sin(t * 0.4) * mid * 2.0  # degrees, keep small

    # =========================
    # 5a. Camera punch (AGGRESSIVE)
    # =========================

    if bass > 0.9:
        zoom *= 1.05 + random.uniform(0.0, 0.08)
        rot += random.uniform(-2.0, 2.0)

    fg = apply_camera(fg, zoom, pan_x, pan_y, rot)

    # Safety guarantee for shape correctness remove after math is debugged
    fg = cv2.resize(fg, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)

    # =========================
    # 5b. Trail
    # =========================
    # Convert fg to float [0,1]
    fg_f = fg.astype(np.float32) / 255.0

    # Update trail: mix current fg into it
    trail_alpha = 0.2  # how much the current frame contributes
    trail = fg_f * trail_alpha + trail * (1.0 - trail_alpha)

    # =========================
    # 5c. Glow/Bloom
    # =========================

    trail = np.minimum(trail, 0.75)

    glow_strength = (
        GLOW_STRENGTH +
        bass * GLOW_AUDIO_BOOST
    ) * CHAOS["glow_bias"]

    trail_pre_glow = trail.copy()
    trail_glow = apply_glow_float(trail_pre_glow, glow_strength)

    # Re-inject current geometry to preserve hue and prevent glow washout
    trail = trail * 0.9 + fg_f * 0.1

    # Reduce blue dominance
    trail[..., 0] *= 0.95  # blue channel (BGR)

    # =========================
    # 6. Composite
    # =========================
    # Background as float
    bg_f = bg.astype(np.float32) / 255.0

    # Composite trails over background
    final_f = np.clip(bg_f + trail_glow, 0, 1)

    # Convert back to uint8 for writing
    final = (final_f * 255).astype(np.uint8)

    # =========================
    # 7. Write
    # =========================
    writer.write(final)

# ===============================
# FINALIZE
# ===============================
writer.release()

# Add audio
subprocess.run([
    "ffmpeg", "-y", "-i", OUT_FILE, "-i", AUDIO_FILE,
    "-c:v", "copy", "-c:a", "aac", "-shortest", FINAL_FILE
], check=True)

print("Done! Video with audio:", FINAL_FILE)
