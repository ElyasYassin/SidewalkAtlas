import asyncio
import os
from datetime import datetime

import numpy as np
from PIL import Image

import omni.usd
import omni.kit.app
import omni.timeline
import omni.replicator.core as rep

from pxr import UsdGeom, Usd
from isaacsim.core.api import World
from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy


# =====================================================================
# CONFIG
# =====================================================================

SPOT_PRIM_PATH = "/World/spot_03"
CAMERA_PRIM_PATH = "/World/spot_03/body/spot_camera"
RESOLUTION = (640, 360)
PHYSICS_RATE = 500

# IMPORTANT:
# Internally:
#   error < 0 means sidewalk is left
#   PID turn < 0 means desired image-space correction left
#
# But Spot policy appears to use opposite yaw sign.
# So final command sent to Spot must invert turn.
INVERT_TURN_COMMAND = True

SETTLE_STEPS = 500
PERCEPTION_EVERY_N_STEPS = 10
DEBUG_SAVE_EVERY_N_STEPS = 500

DEBUG_DIR = "C:/Users/Elyas/sidewalk_dataset/walk_debug"
os.makedirs(DEBUG_DIR, exist_ok=True)

# Walking
NOMINAL_FORWARD_SPEED = 0.28
MIN_FORWARD_SPEED = 0.08
WALK_DURATION_STEPS = 100000  # 100000 / 500 = 200 seconds

# PID / PD steering controller
KP_TURN = 0.60
KI_TURN = 0.00
KD_TURN = 0.04

MAX_TURN_CMD = 0.55
MAX_INTEGRAL_ERROR = 1.0
DERIVATIVE_FILTER_ALPHA = 0.25

# Recovery behavior
BACKUP_SPEED = -0.10
BACKUP_STEPS = 350
SCAN_STEPS = 500
SEARCH_TURN_RATE = 0.45

# Reacquire behavior
REACQUIRE_NEAR_PIXELS = 5000
REACQUIRE_TOTAL_PIXELS = 10000
MAX_RECOVERY_CYCLES = 4
STOP_AFTER_DONE = True

# Perception thresholds
MIN_MASK_PIXELS = 300
MIN_NEAR_PIXELS = 800

# Diagnostic
FORCE_FORWARD = False

# Classes that count as walkable for navigation
WALKABLE_CLASSES = {"sidewalk", "defect"}


# =====================================================================
# RGB + DEPTH CAPTURE CONFIG
# =====================================================================

CAPTURE_SENSOR_DATA = True

# 50 steps / 500 Hz = 10 FPS.
# Use 100 for 5 FPS, 250 for 2 FPS, etc.
CAPTURE_EVERY_N_STEPS = 1050

CAPTURE_DIR = "C:/Users/Elyas/sidewalk_dataset/walk_captures"

RUN_NAME = datetime.now().strftime("run_%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(CAPTURE_DIR, RUN_NAME)
RGB_DIR = os.path.join(RUN_DIR, "rgb")
DEPTH_NPY_DIR = os.path.join(RUN_DIR, "depth_npy")
DEPTH_VIS_DIR = os.path.join(RUN_DIR, "depth_vis")

for d in [RUN_DIR, RGB_DIR, DEPTH_NPY_DIR, DEPTH_VIS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Saving walking RGB/depth data to: {RUN_DIR}")


# =====================================================================
# SEMANTIC TAGGING
# =====================================================================

def tag_walkables():
    """Apply semantic class labels to walkable prims."""
    stage = omni.usd.get_context().get_stage()

    try:
        from isaacsim.core.utils.semantics import add_update_semantics

        def tag(prim, class_name):
            add_update_semantics(prim, class_name)

    except ImportError:
        from pxr import Semantics

        def tag(prim, class_name):
            sem_api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
            sem_api.CreateSemanticTypeAttr("class")
            sem_api.CreateSemanticDataAttr(class_name)

    DEFECTS = [
        "/World/assembly_City/collapsed_1",
        "/World/assembly_City/collapsed_2",
        "/World/assembly_City/collapsed_3",
        "/World/assembly_City/collapsed_4",
        "/World/assembly_City/crack_1",
        "/World/assembly_City/crack_2",
        "/World/assembly_City/crack_3",
        "/World/assembly_City/crack_4",
        "/World/assembly_City/pothole_1",
        "/World/assembly_City/pothole_2",
        "/World/assembly_City/uneven_1",
        "/World/assembly_City/uneven_2",
        "/World/assembly_City/uneven_3",
        "/World/assembly_City/uneven_4",
        "/World/assembly_City/uneven_5",
        "/World/assembly_City/uneven_6",
        "/World/assembly_City/uneven_7",
        "/World/assembly_City/Meshy_AI_sidewalk_0404170947_texture",
    ]

    CURBS = [
        "/World/assembly_City/Meshy_AI_Curved_curb_along_a_q_0521183005_texture",
        "/World/assembly_City/Meshy_AI_Yellow_curb_beside_th_0521183948_texture",
        "/World/assembly_City/Meshy_AI_Mossy_Curb_Along_the__0521183157_texture",
    ]

    tagged = 0

    for paths, cls in [
        (DEFECTS, "defect"),
        (CURBS, "curb"),
    ]:
        for p in paths:
            prim = stage.GetPrimAtPath(p)

            if prim.IsValid():
                tag(prim, cls)
                tagged += 1
            else:
                print(f"⚠ Prim not found for semantic tag: {p}")

    print(f"✓ Tagged {tagged} prims with semantic classes")


# =====================================================================
# STATE
# =====================================================================

_state = {
    "spot": None,
    "render_product": None,
    "rgb_annotator": None,
    "semantic_annotator": None,
    "depth_annotator": None,

    "physics_ready": False,
    "step_count": 0,
    "current_command": np.array([0.0, 0.0, 0.0]),

    "last_perception_step": 0,
    "last_save_step": 0,
    "last_capture_step": 0,
    "capture_frame_idx": 0,

    "last_debug": None,
    "walkable_ids": set(),

    # Navigation memory
    "last_turn_direction": 1.0,
    "last_sidewalk_error": 0.0,

    # Command smoothing
    "last_forward": 0.0,
    "last_turn": 0.0,

    # PID state
    "pid_prev_error": 0.0,
    "pid_integral_error": 0.0,
    "pid_filtered_derivative": 0.0,

    # Recovery state
    "recovery_mode": None,
    "recovery_until_step": 0,
    "recovery_reason": "",
    "recovery_cycles": 0,

    "done": False,
}


# =====================================================================
# PERCEPTION
# =====================================================================

def get_walkable_ids_from_info(info):
    """Extract semantic IDs that match our walkable classes."""
    ids = set()
    id_to_labels = info.get("idToLabels", {}) if isinstance(info, dict) else {}

    for id_str, label_info in id_to_labels.items():
        if isinstance(label_info, dict):
            class_name = label_info.get("class", "").lower()
        else:
            class_name = str(label_info).lower()

        if class_name in WALKABLE_CLASSES:
            try:
                ids.add(int(id_str))
            except (ValueError, TypeError):
                pass

    return ids


def compute_sidewalk_mask():
    """Get walkable mask from semantic segmentation."""
    sem_data = _state["semantic_annotator"].get_data()

    if not isinstance(sem_data, dict) or "data" not in sem_data:
        return None

    sem_map = sem_data["data"]

    if sem_map is None or sem_map.size == 0:
        return None

    info = sem_data.get("info", {})
    ids = get_walkable_ids_from_info(info)
    _state["walkable_ids"] = ids

    if not ids:
        return np.zeros_like(sem_map, dtype=bool)

    mask = np.zeros_like(sem_map, dtype=bool)

    for wid in ids:
        mask |= sem_map == wid

    return mask


def make_semantic_id_image():
    """
    Visualize raw semantic segmentation IDs as colors.
    Each semantic ID gets a deterministic pseudo-random color.
    """
    sem_data = _state["semantic_annotator"].get_data()

    if not isinstance(sem_data, dict) or "data" not in sem_data:
        return None

    sem_map = sem_data["data"]

    if sem_map is None or sem_map.size == 0:
        return None

    sem_map = sem_map.astype(np.int32)

    h, w = sem_map.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    unique_ids = np.unique(sem_map)

    for sid in unique_ids:
        if sid == 0:
            color = np.array([0, 0, 0], dtype=np.uint8)
        else:
            rng = np.random.default_rng(int(sid))
            color = rng.integers(40, 255, size=3, dtype=np.uint8)

        color_img[sem_map == sid] = color

    return color_img


def print_semantic_info():
    """Print Isaac semantic info mapping from IDs to labels."""
    sem_data = _state["semantic_annotator"].get_data()

    if isinstance(sem_data, dict):
        info = sem_data.get("info", {})
        print("SEMANTIC INFO:", info)
    else:
        print("SEMANTIC DATA TYPE:", type(sem_data))


def print_semantic_counts():
    """Print semantic IDs and pixel counts."""
    sem_data = _state["semantic_annotator"].get_data()

    if not isinstance(sem_data, dict) or "data" not in sem_data:
        print("No semantic data")
        return

    sem_map = sem_data["data"]
    info = sem_data.get("info", {})

    unique_ids, counts = np.unique(sem_map, return_counts=True)

    print("\n=== SEMANTIC COUNTS ===")
    print("INFO:", info)

    id_to_labels = info.get("idToLabels", {})

    for sid, count in zip(unique_ids, counts):
        sid_int = int(sid)

        label = None
        if str(sid_int) in id_to_labels:
            label = id_to_labels[str(sid_int)]
        elif sid_int in id_to_labels:
            label = id_to_labels[sid_int]

        print(f"ID {sid_int}: pixels={int(count)}, label={label}")


# =====================================================================
# RGB + DEPTH CAPTURE HELPERS
# =====================================================================

def rgb_to_uint8(rgb):
    """Convert Replicator RGB output to uint8 RGB."""
    if rgb is None:
        return None

    arr = np.array(rgb)

    if arr.size == 0:
        return None

    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)

    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]

    return arr


def extract_depth_array(depth_data):
    """
    Handle Replicator depth annotator output.
    It may return an ndarray or a dict containing 'data'.
    """
    if depth_data is None:
        return None

    if isinstance(depth_data, dict):
        depth = depth_data.get("data", None)
    else:
        depth = depth_data

    if depth is None:
        return None

    depth = np.array(depth).astype(np.float32)

    if depth.size == 0:
        return None

    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    return depth


def make_depth_visualization(depth_m, max_depth_m=10.0):
    """
    Convert depth in meters to uint8 grayscale visualization.
    Near = brighter, far = darker.
    """
    if depth_m is None:
        return None

    depth = depth_m.copy()

    valid = np.isfinite(depth) & (depth > 0)

    if not valid.any():
        return np.zeros(depth.shape, dtype=np.uint8)

    clipped = np.clip(depth, 0.0, max_depth_m)

    vis = 255.0 * (1.0 - clipped / max_depth_m)
    vis[~valid] = 0

    return vis.astype(np.uint8)


def save_rgb_depth_frame():
    """
    Save synchronized RGB and depth frames from Spot camera.

    Saves:
    - rgb/rgb_000001_step_0000600.png
    - depth_npy/depth_000001_step_0000600.npy
    - depth_vis/depth_vis_000001_step_0000600.png
    """
    if not CAPTURE_SENSOR_DATA:
        return

    rgb_data = _state["rgb_annotator"].get_data()
    depth_data = _state["depth_annotator"].get_data()

    rgb = rgb_to_uint8(rgb_data)
    depth_m = extract_depth_array(depth_data)

    if rgb is None or depth_m is None:
        print("⚠ Could not save RGB/depth frame: missing data")
        return

    frame_idx = _state["capture_frame_idx"]
    sc = _state["step_count"]

    rgb_path = os.path.join(
        RGB_DIR,
        f"rgb_{frame_idx:06d}_step_{sc:07d}.png",
    )
    depth_npy_path = os.path.join(
        DEPTH_NPY_DIR,
        f"depth_{frame_idx:06d}_step_{sc:07d}.npy",
    )
    depth_vis_path = os.path.join(
        DEPTH_VIS_DIR,
        f"depth_vis_{frame_idx:06d}_step_{sc:07d}.png",
    )

    Image.fromarray(rgb).save(rgb_path)
    np.save(depth_npy_path, depth_m)

    depth_vis = make_depth_visualization(depth_m, max_depth_m=10.0)
    Image.fromarray(depth_vis).save(depth_vis_path)

    _state["capture_frame_idx"] += 1


# =====================================================================
# CONTROL HELPERS
# =====================================================================

def apply_turn_sign(turn):
    """
    Convert internal image-space turn to Spot command turn.
    """
    if INVERT_TURN_COMMAND:
        return -turn
    return turn


def make_command(forward, turn):
    """
    Build final command sent to Spot.
    This is the ONLY place where turn sign should be inverted.
    """
    spot_turn = apply_turn_sign(turn)
    return np.array([forward, 0.0, spot_turn])


def smooth_command(forward, turn, alpha=0.35):
    """
    Smooth forward and internal turn commands.
    Turn is still image-space/internal here, before Spot sign inversion.
    """
    prev_forward = _state["last_forward"]
    prev_turn = _state["last_turn"]

    smoothed_forward = alpha * forward + (1.0 - alpha) * prev_forward
    smoothed_turn = alpha * turn + (1.0 - alpha) * prev_turn

    _state["last_forward"] = smoothed_forward
    _state["last_turn"] = smoothed_turn

    return smoothed_forward, smoothed_turn


def reset_pid():
    """Reset PID state."""
    _state["pid_prev_error"] = 0.0
    _state["pid_integral_error"] = 0.0
    _state["pid_filtered_derivative"] = 0.0


def compute_pid_turn(error, dt):
    """
    PID controller for steering.

    error < 0 means sidewalk center is left.
    error > 0 means sidewalk center is right.
    """
    prev_error = _state["pid_prev_error"]
    integral = _state["pid_integral_error"]
    filtered_derivative = _state["pid_filtered_derivative"]

    derivative = (error - prev_error) / max(dt, 1e-6)

    filtered_derivative = (
        (1.0 - DERIVATIVE_FILTER_ALPHA) * filtered_derivative
        + DERIVATIVE_FILTER_ALPHA * derivative
    )

    integral += error * dt
    integral = float(np.clip(integral, -MAX_INTEGRAL_ERROR, MAX_INTEGRAL_ERROR))

    turn = (
        KP_TURN * error
        + KI_TURN * integral
        + KD_TURN * filtered_derivative
    )

    turn = float(np.clip(turn, -MAX_TURN_CMD, MAX_TURN_CMD))

    _state["pid_prev_error"] = error
    _state["pid_integral_error"] = integral
    _state["pid_filtered_derivative"] = filtered_derivative

    return turn


def get_recovery_turn_direction():
    """
    Return internal image-space direction toward where sidewalk was last seen.
    Negative = sidewalk was left.
    Positive = sidewalk was right.
    """
    last_error = _state.get("last_sidewalk_error", 0.0)

    if abs(last_error) > 0.05:
        return float(np.sign(last_error))

    return float(_state.get("last_turn_direction", 1.0))


def start_recovery(reason, scan_direction=None):
    """
    Start recovery behavior:
    1. Back up.
    2. Then back up while turning toward last known sidewalk direction.
    """
    sc = _state["step_count"]

    if _state["recovery_mode"] is not None:
        return

    if scan_direction is not None and scan_direction != 0:
        _state["last_turn_direction"] = float(np.sign(scan_direction))

    _state["recovery_mode"] = "backup"
    _state["recovery_until_step"] = sc + BACKUP_STEPS
    _state["recovery_reason"] = reason


# =====================================================================
# NAVIGATION
# =====================================================================

def compute_velocity_command(mask):
    """
    PID-based sidewalk-following controller.
    """

    sc = _state["step_count"]

    # ---------------------------------------------------------------
    # Active recovery behavior
    # ---------------------------------------------------------------
    if _state["recovery_mode"] is not None:
        reacquired = False

        if mask is not None and mask.sum() >= REACQUIRE_TOTAL_PIXELS:
            h, w = mask.shape

            near_band = mask[2 * h // 3 :, :]
            lower_center = mask[2 * h // 3 :, w // 3 : 2 * w // 3]

            near_pixels = int(near_band.sum())
            lower_center_pixels = int(lower_center.sum())

            if near_pixels >= REACQUIRE_NEAR_PIXELS and lower_center_pixels > 1200:
                reacquired = True

        if reacquired:
            _state["recovery_mode"] = None
            _state["recovery_reason"] = ""
            _state["recovery_cycles"] = 0
            reset_pid()

        else:
            if sc < _state["recovery_until_step"]:
                if _state["recovery_mode"] == "backup":
                    forward = BACKUP_SPEED
                    turn = 0.0
                    forward, turn = smooth_command(forward, turn, alpha=0.6)
                    return make_command(forward, turn), 0, 0, 0

                if _state["recovery_mode"] == "scan":
                    forward = BACKUP_SPEED
                    turn = SEARCH_TURN_RATE * get_recovery_turn_direction()
                    forward, turn = smooth_command(forward, turn, alpha=0.6)
                    return make_command(forward, turn), 0, 0, 0

            else:
                if _state["recovery_mode"] == "backup":
                    _state["recovery_mode"] = "scan"
                    _state["recovery_until_step"] = sc + SCAN_STEPS

                    forward = BACKUP_SPEED
                    turn = SEARCH_TURN_RATE * get_recovery_turn_direction()
                    forward, turn = smooth_command(forward, turn, alpha=0.6)
                    return make_command(forward, turn), 0, 0, 0

                if _state["recovery_mode"] == "scan":
                    _state["recovery_cycles"] += 1

                    if _state["recovery_cycles"] >= MAX_RECOVERY_CYCLES:
                        _state["last_turn_direction"] *= -1.0
                        _state["last_sidewalk_error"] *= -1.0
                        _state["recovery_cycles"] = 0
                        _state["recovery_mode"] = "backup"
                        _state["recovery_until_step"] = sc + int(BACKUP_STEPS * 2.0)
                        _state["recovery_reason"] = "repeated scan failure"
                        reset_pid()

                        forward = BACKUP_SPEED
                        turn = 0.0
                        forward, turn = smooth_command(forward, turn, alpha=0.6)
                        return make_command(forward, turn), 0, 0, 0

                    _state["recovery_mode"] = "scan"
                    _state["recovery_until_step"] = sc + SCAN_STEPS

                    forward = BACKUP_SPEED
                    turn = SEARCH_TURN_RATE * get_recovery_turn_direction()
                    forward, turn = smooth_command(forward, turn, alpha=0.6)
                    return make_command(forward, turn), 0, 0, 0

    # ---------------------------------------------------------------
    # No sidewalk visible
    # ---------------------------------------------------------------
    if mask is None or mask.sum() < MIN_MASK_PIXELS:
        start_recovery("no sidewalk visible")
        reset_pid()

        forward = BACKUP_SPEED
        turn = 0.0
        forward, turn = smooth_command(forward, turn, alpha=0.6)
        return make_command(forward, turn), 0, 0, 0

    h, w = mask.shape

    # ---------------------------------------------------------------
    # Regions
    # ---------------------------------------------------------------
    near_band = mask[2 * h // 3 :, :]
    lookahead = mask[h // 2 : 5 * h // 6, :]

    left = int(lookahead[:, : w // 3].sum())
    center = int(lookahead[:, w // 3 : 2 * w // 3].sum())
    right = int(lookahead[:, 2 * w // 3 :].sum())

    near_left = int(near_band[:, : w // 3].sum())
    near_center = int(near_band[:, w // 3 : 2 * w // 3].sum())
    near_right = int(near_band[:, 2 * w // 3 :].sum())
    near_total = near_left + near_center + near_right

    if near_total < MIN_NEAR_PIXELS:
        start_recovery("too little near sidewalk")
        reset_pid()

        forward = BACKUP_SPEED
        turn = 0.0
        forward, turn = smooth_command(forward, turn, alpha=0.6)
        return make_command(forward, turn), left, center, right

    # ---------------------------------------------------------------
    # Estimate sidewalk centerline from near-band columns
    # ---------------------------------------------------------------
    col_counts = near_band.sum(axis=0)

    col_threshold = max(3, int(0.08 * near_band.shape[0]))
    walkable_cols = col_counts > col_threshold

    if not walkable_cols.any():
        start_recovery("no walkable columns in near band")
        reset_pid()

        forward = BACKUP_SPEED
        turn = 0.0
        forward, turn = smooth_command(forward, turn, alpha=0.6)
        return make_command(forward, turn), left, center, right

    intervals = []
    in_segment = False
    start = 0

    for i, val in enumerate(walkable_cols):
        if val and not in_segment:
            start = i
            in_segment = True
        elif not val and in_segment:
            intervals.append((start, i - 1))
            in_segment = False

    if in_segment:
        intervals.append((start, w - 1))

    image_center = w / 2.0

    best_interval = None
    best_score = float("inf")

    for a, b in intervals:
        interval_width = b - a + 1
        interval_center = (a + b) / 2.0

        score = abs(interval_center - image_center) - 0.25 * interval_width

        if score < best_score:
            best_score = score
            best_interval = (a, b)

    if best_interval is None:
        start_recovery("no valid sidewalk interval")
        reset_pid()

        forward = BACKUP_SPEED
        turn = 0.0
        forward, turn = smooth_command(forward, turn, alpha=0.6)
        return make_command(forward, turn), left, center, right

    a, b = best_interval
    sidewalk_center_x = (a + b) / 2.0
    sidewalk_width = b - a + 1

    error = (sidewalk_center_x - image_center) / image_center
    error = float(np.clip(error, -1.0, 1.0))

    _state["last_sidewalk_error"] = error

    if abs(error) > 0.05:
        _state["last_turn_direction"] = float(np.sign(error))

    # ---------------------------------------------------------------
    # PID / PD steering
    # ---------------------------------------------------------------
    dt = PERCEPTION_EVERY_N_STEPS / PHYSICS_RATE

    DEAD_BAND = 0.04
    pid_error = 0.0 if abs(error) < DEAD_BAND else error

    turn = compute_pid_turn(pid_error, dt)

    # ---------------------------------------------------------------
    # Forward speed scheduling
    # ---------------------------------------------------------------
    center_ratio = near_center / max(near_total, 1)
    abs_error = abs(error)

    lower_center_weak = center_ratio < 0.20

    if lower_center_weak:
        forward = 0.0

    elif abs_error < 0.10 and sidewalk_width > 0.40 * w:
        forward = NOMINAL_FORWARD_SPEED

    elif abs_error < 0.25:
        forward = 0.18

    elif abs_error < 0.45:
        forward = 0.08

    else:
        forward = 0.0

    if abs(turn) > 0.35:
        forward = min(forward, 0.06)

    if forward > 0.10:
        turn = float(np.clip(turn, -0.25, 0.25))

    forward, turn = smooth_command(forward, turn, alpha=0.45)

    return make_command(forward, turn), left, center, right


# =====================================================================
# DEBUG VISUALIZATION
# =====================================================================

def make_debug_image(rgb, mask):
    """
    Make side-by-side image:
    left = RGB
    right = RGB darkened with walkable mask highlighted green
    """
    if rgb is None:
        return None

    rgb = rgb_to_uint8(rgb)

    if rgb is None:
        return None

    overlay = rgb.copy().astype(np.float32)

    if mask is not None:
        if mask.shape != rgb.shape[:2]:
            from PIL import Image as PILImage

            mask_img = PILImage.fromarray(mask.astype(np.uint8) * 255)
            mask_img = mask_img.resize((rgb.shape[1], rgb.shape[0]))
            mask = np.array(mask_img) > 127

        overlay[~mask] *= 0.4
        overlay[mask, 1] = np.minimum(overlay[mask, 1] + 80, 255)

    overlay = overlay.astype(np.uint8)

    def add_lines(img):
        img = img.copy()
        h, w = img.shape[:2]

        img[:, w // 3 - 1 : w // 3 + 1] = [255, 0, 0]
        img[:, 2 * w // 3 - 1 : 2 * w // 3 + 1] = [255, 0, 0]

        img[h // 2 - 1 : h // 2 + 1, :] = [255, 0, 0]

        y1 = h // 2
        y2 = 5 * h // 6
        img[y1 - 1 : y1 + 1, :] = [255, 255, 0]
        img[y2 - 1 : y2 + 1, :] = [255, 255, 0]

        near_y = 2 * h // 3
        img[near_y - 1 : near_y + 1, :] = [0, 255, 255]

        return img

    return np.hstack([add_lines(rgb), add_lines(overlay)])


# =====================================================================
# PHYSICS CALLBACK
# =====================================================================

def on_physics_step(step_size):
    if not _state["physics_ready"]:
        _state["physics_ready"] = True
        _state["spot"].initialize()
        _state["spot"].post_reset()
        _state["spot"].robot.set_joints_default_state(_state["spot"].default_pos)
        print(">>> Spot policy initialized.")
        return

    if _state.get("done", False) and STOP_AFTER_DONE:
        _state["spot"].forward(step_size, np.array([0.0, 0.0, 0.0]))
        _state["step_count"] += 1
        return

    sc = _state["step_count"]

    if sc % 500 == 0:
        print(
            f">>> Physics step {sc} "
            f"(t={sc / PHYSICS_RATE:.1f}s), "
            f"cmd={_state['current_command']}"
        )

    if sc < SETTLE_STEPS:
        _state["spot"].forward(step_size, np.array([0.0, 0.0, 0.0]))
    else:
        _state["spot"].forward(step_size, _state["current_command"])

    _state["step_count"] += 1


# =====================================================================
# ORIENTATION CHECK
# =====================================================================

def check_spot_orientation():
    stage = omni.usd.get_context().get_stage()
    spot_prim = stage.GetPrimAtPath(SPOT_PRIM_PATH)
    body_prim = stage.GetPrimAtPath(f"{SPOT_PRIM_PATH}/body")

    if not spot_prim.IsValid():
        print("Spot prim not found!")
        return

    m = UsdGeom.Xformable(spot_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    pos = m.ExtractTranslation()

    if body_prim.IsValid():
        body_m = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        up = body_m.TransformDir((0, 0, 1))

        print("\n=== Orientation check ===")
        print(f"Spot position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        print(f"Body up-vector: ({up[0]:.3f}, {up[1]:.3f}, {up[2]:.3f})")

        if up[2] > 0.85:
            print("✓ Spot is upright")
        elif up[2] < 0.3:
            print("✗ Spot has fallen; body up-vector is sideways or down")
        else:
            print(f"⚠ Spot is tilted; up-Z = {up[2]:.2f}")


# =====================================================================
# MAIN
# =====================================================================

async def run_demo():
    try:
        # Reset run state
        _state["done"] = False
        _state["step_count"] = 0
        _state["current_command"] = np.array([0.0, 0.0, 0.0])
        _state["last_forward"] = 0.0
        _state["last_turn"] = 0.0
        _state["last_perception_step"] = 0
        _state["last_save_step"] = 0
        _state["last_capture_step"] = 0
        _state["capture_frame_idx"] = 0
        _state["recovery_mode"] = None
        _state["recovery_reason"] = ""
        _state["recovery_cycles"] = 0
        reset_pid()

        stage = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath(CAMERA_PRIM_PATH)

        if cam_prim.IsValid():
            cam_geom = UsdGeom.Camera(cam_prim)
            cam_geom.GetFocalLengthAttr().Set(2.2)
            cam_geom.GetHorizontalApertureAttr().Set(5.0)
            cam_geom.GetVerticalApertureAttr().Set(2.8)
            cam_geom.GetClippingRangeAttr().Set((0.05, 100.0))
            print("✓ Camera intrinsics set")
        else:
            print(f"⚠ Camera prim not found at {CAMERA_PRIM_PATH}")

        tag_walkables()

        World.clear_instance()

        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / PHYSICS_RATE,
            rendering_dt=10.0 / PHYSICS_RATE,
        )

        await world.initialize_simulation_context_async()
        print("✓ Simulation context initialized")

        _state["spot"] = SpotFlatTerrainPolicy(
            prim_path=SPOT_PRIM_PATH,
            name="Spot",
        )
        print("✓ Spot wrapped")

        for _ in range(60):
            await omni.kit.app.get_app().next_update_async()

        await world.reset_async()
        print("✓ World reset")

        _state["render_product"] = rep.create.render_product(
            CAMERA_PRIM_PATH,
            resolution=RESOLUTION,
        )
        print("✓ Render product created")

        # RGB annotator
        _state["rgb_annotator"] = rep.AnnotatorRegistry.get_annotator("rgb")
        _state["rgb_annotator"].attach(_state["render_product"])

        # Semantic annotator
        _state["semantic_annotator"] = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation"
        )
        _state["semantic_annotator"].attach(_state["render_product"])

        # Depth annotator
        _state["depth_annotator"] = rep.AnnotatorRegistry.get_annotator(
            "distance_to_image_plane"
        )
        _state["depth_annotator"].attach(_state["render_product"])

        print("✓ RGB + Semantic + Depth annotators attached")

        world.add_physics_callback(
            "physics_step",
            callback_fn=on_physics_step,
        )
        print("✓ Physics callback registered")

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

        for _ in range(60):
            await omni.kit.app.get_app().next_update_async()

        print(f"✓ Timeline playing = {timeline.is_playing()}")

        print_semantic_info()
        print_semantic_counts()

        print(f"\n=== Loop starting. step_count={_state['step_count']} ===\n")

        last_log_step = 0
        target_steps = SETTLE_STEPS + WALK_DURATION_STEPS

        print(
            f"TARGET STEPS = {target_steps}, "
            f"current step_count = {_state['step_count']}, "
            f"walk seconds = {WALK_DURATION_STEPS / PHYSICS_RATE:.1f}"
        )

        while _state["step_count"] < target_steps:
            await omni.kit.app.get_app().next_update_async()

            sc = _state["step_count"]

            if sc < SETTLE_STEPS:
                continue

            # Perception
            if sc - _state["last_perception_step"] >= PERCEPTION_EVERY_N_STEPS:
                _state["last_perception_step"] = sc

                try:
                    mask = compute_sidewalk_mask()
                    cmd, l, c, r = compute_velocity_command(mask)

                    if FORCE_FORWARD:
                        cmd = np.array([NOMINAL_FORWARD_SPEED, 0.0, 0.0])

                    _state["current_command"] = cmd

                    rgb_data = _state["rgb_annotator"].get_data()
                    _state["last_debug"] = (rgb_data, mask, l, c, r)

                    if sc - last_log_step >= 100:
                        last_log_step = sc

                        print(
                            f"  t={sc / PHYSICS_RATE:.1f}s  "
                            f"L/C/R={l}/{c}/{r}  "
                            f"walkable_ids={_state['walkable_ids']}  "
                            f"last_turn_dir={_state['last_turn_direction']:+.0f}  "
                            f"last_sidewalk_error={_state['last_sidewalk_error']:+.2f}  "
                            f"pid_prev_error={_state['pid_prev_error']:+.2f}  "
                            f"pid_deriv={_state['pid_filtered_derivative']:+.2f}  "
                            f"internal_turn={_state['last_turn']:+.2f}  "
                            f"spot_turn={cmd[2]:+.2f}  "
                            f"recovery={_state['recovery_mode']}  "
                            f"cycles={_state['recovery_cycles']}  "
                            f"reason={_state['recovery_reason']}  "
                            f"cmd=({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f})"
                        )

                except Exception as e:
                    print(f"Perception error: {e}")

            # RGB/depth capture
            if (
                CAPTURE_SENSOR_DATA
                and sc >= SETTLE_STEPS
                and sc - _state["last_capture_step"] >= CAPTURE_EVERY_N_STEPS
            ):
                _state["last_capture_step"] = sc

                try:
                    save_rgb_depth_frame()
                except Exception as e:
                    print(f"RGB/depth capture error: {e}")

            # Debug save
            if (
                sc - _state["last_save_step"] >= DEBUG_SAVE_EVERY_N_STEPS
                and _state["last_debug"] is not None
            ):
                _state["last_save_step"] = sc

                rgb, mask, l, c, r = _state["last_debug"]

                try:
                    if rgb is not None and np.array(rgb).size > 0:
                        combined = make_debug_image(rgb, mask)
                        semantic_img = make_semantic_id_image()

                        if combined is not None:
                            path = os.path.join(
                                DEBUG_DIR,
                                f"frame_t{sc // PHYSICS_RATE:04d}s_rgb_mask.png",
                            )
                            Image.fromarray(combined).save(path)

                        if semantic_img is not None:
                            sem_path = os.path.join(
                                DEBUG_DIR,
                                f"frame_t{sc // PHYSICS_RATE:04d}s_semantic_ids.png",
                            )
                            Image.fromarray(semantic_img).save(sem_path)

                        print_semantic_info()
                        print_semantic_counts()

                except Exception as e:
                    print(f"Debug save error: {e}")

        _state["done"] = True
        _state["current_command"] = np.array([0.0, 0.0, 0.0])

        print("\nDone walking.")
        print(f"Captured {_state['capture_frame_idx']} RGB/depth frames.")
        print(f"Saved capture run to: {RUN_DIR}")

        check_spot_orientation()

    except Exception as e:
        print(f"ERROR: {e}")

        import traceback
        traceback.print_exc()


# =====================================================================
# TASK GUARD
# =====================================================================

try:
    if "_sidewalk_nav_task" in globals():
        if _sidewalk_nav_task is not None and not _sidewalk_nav_task.done():
            _sidewalk_nav_task.cancel()
            print("Cancelled previous sidewalk navigation task.")
except Exception as e:
    print(f"Could not cancel previous task: {e}")

_sidewalk_nav_task = asyncio.ensure_future(run_demo())
print("Walk + semantic perception navigation demo with RGB/depth capture scheduled.")