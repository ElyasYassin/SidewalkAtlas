import asyncio
import csv
import json
import os
from datetime import datetime

import numpy as np
from PIL import Image

import omni.usd
import omni.kit.app
import omni.timeline
import omni.replicator.core as rep

from pxr import UsdGeom, Usd, PhysxSchema
from omni.physx import get_physx_scene_query_interface
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

# Live perception window (omni.ui inside Isaac Sim):
# RGB | walkable mask overlay | binary sidewalk / not-sidewalk.
# Visualization only -- does not touch navigation.
LIVE_VIEW_ENABLED = True
LIVE_VIEW_EVERY_N_STEPS = 40      # ~10 updates/sec at 500 Hz
LIVE_VIEW_MAX_HEIGHT = 240        # downscale composite before pushing, to stay cheap
SIDEWALK_COLOR = (0, 200, 0)
NOT_SIDEWALK_COLOR = (55, 55, 55)
DEPTH_VIEW_MAX_M = 10.0           # depth colormap range (m)

# Coverage: stamp a swath ahead of Spot into a world grid as it walks, and
# show a live top-down Coverage Map (green = swept, blue = path).
COVERAGE_ENABLED = True
COVERAGE_CELL_M = 0.05            # 5 cm world cells
# Swath = the corridor Spot is considered to "cover" as it passes. Made wide
# (approaching real sidewalk width) and spanning from behind the body to ahead
# so consecutive stamps merge into one continuous, fuller band on the map.
COVERAGE_SWATH_WIDTH_M = 1.6      # cross-track footprint (m)
COVERAGE_SWATH_LEN_M = 1.6        # along-track footprint (m)
COVERAGE_SWATH_AHEAD_M = -0.6     # swath starts this far ahead of body (negative = behind)
COVERAGE_MARK_EVERY_N_STEPS = 12  # how often to stamp the swath (denser = fewer gaps)
COVERAGE_MAP_PATH = os.path.join(DEBUG_DIR, "coverage_map.png")

# Coverage-seeking weave: instead of tracking only the centerline, zig-zag
# across the sidewalk width so Spot actually covers the whole surface. It aims
# toward one edge, and flips to the other edge once its near footprint reaches
# the current side. Set False for plain centerline-following.
COVERAGE_SWEEP_ENABLED = True
SWEEP_FRACTION = 0.7              # aim this far from center toward the edge (0..1)
SWEEP_EDGE_FRAC = 0.15           # "reached edge" when that side's near band < this*total

# Walking
# The Spot policy was trained with lin_vel_x command range [-2.0, 3.0] m/s
# (from spot_env.yaml). 0.28 is only ~9% of max -> a barely-visible crawl that
# looks frozen. Command a real fraction of the range so it actually strides.
NOMINAL_FORWARD_SPEED = 1.0
MIN_FORWARD_SPEED = 0.3
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

# Classes that count as walkable for navigation.
# Binary segmentation: every walkable surface is tagged "sidewalk".
WALKABLE_CLASSES = {"sidewalk"}


# =====================================================================
# ONLINE SIDEWALK EVENNESS DETECTION
# =====================================================================

# Samples the collision surface directly beneath Spot while it walks, then
# computes detrended height variation over a sliding distance window. Because
# collision fidelity was validated separately, this profile is what Spot's feet
# physically encounter. Body vertical acceleration is logged as a secondary
# traversability signal.
EVENNESS_ENABLED = True
EVENNESS_SAMPLE_EVERY_N_STEPS = 5      # 100 Hz at 500 Hz physics
EVENNESS_AUTOSAVE_EVERY_N_SAMPLES = 500  # rewrite CSV every ~5 simulated seconds
EVENNESS_WINDOW_M = 1.0                # rolling distance window
EVENNESS_MIN_WINDOW_M = 0.25           # do not classify tiny windows
EVENNESS_MIN_SAMPLES = 20
EVENNESS_RAY_HEIGHT_M = 1.5
EVENNESS_RAY_MAX_DIST_M = 4.0
EVENNESS_IGNORE_ROOTS = (SPOT_PRIM_PATH,)

# Peak-to-peak height thresholds after removing local sidewalk slope.
# Tune these to your study definition.
EVENNESS_SMOOTH_MAX_MM = 3.0
EVENNESS_MILD_MAX_MM = 10.0
EVENNESS_UNEVEN_MAX_MM = 25.0

EVENNESS_CSV_PATH = os.path.join(RUN_DIR, "sidewalk_evenness_timeseries.csv") if "RUN_DIR" in globals() else None
EVENNESS_SUMMARY_PATH = os.path.join(RUN_DIR, "sidewalk_evenness_summary.json") if "RUN_DIR" in globals() else None


# =====================================================================
# PROPRIOCEPTIVE SENSING
# =====================================================================

# The model that has to run on a real Spot cannot see the terrain -- it only
# feels it. These are the signals it will feel, logged into the same row as
# the raycast ground truth so sensor and label are synchronised by
# construction.
#
# Ordered by how well each is expected to survive sim-to-real:
#
#   foot contact + touchdown height  geometry. The foot has to end up where
#       the ground is, whoever scheduled the step. Mirrors the real SDK's
#       foot_state[].
#   joint position / velocity / effort  partly controller-dependent: effort
#       especially encodes HOW the controller rejected the disturbance, and
#       the real robot runs a different controller.
#   body IMU  last. The body is the most stabilised point on the robot, so
#       it sees terrain only after the controller has attenuated it.
#
# Prim names confirmed against this Spot asset by probe_spot_state.py:
#   feet   fl_foot, fr_foot, hl_foot, hr_foot
#   joints {fl,fr,hl,hr}_{hx,hy,kn}  (hip-x abduction, hip-y flexion, knee)
# Joint ORDER is resolved from the articulation at runtime, not assumed --
# USD traversal order and articulation DOF order are not the same thing.

SENSORS_ENABLED = True

FOOT_NAMES = ("fl_foot", "fr_foot", "hl_foot", "hr_foot")
JOINT_NAMES = (
    "fl_hx", "fl_hy", "fl_kn",
    "fr_hx", "fr_hy", "fr_kn",
    "hl_hx", "hl_hy", "hl_kn",
    "hr_hx", "hr_hy", "hr_kn",
)

IMU_ENABLED = True
IMU_PRIM_PATH = f"{SPOT_PRIM_PATH}/body/proprio_imu"

CONTACT_SENSORS_ENABLED = True
CONTACT_FORCE_THRESHOLD_N = 1.0     # below this the foot reads as airborne

# Sampling rate note: sensors ride on EVENNESS_SAMPLE_EVERY_N_STEPS (5 steps =
# 100 Hz). A real Spot streams robot state closer to 30-50 Hz, and the plan is
# to match it -- but the matching is done by DECIMATING at training time, not
# by logging coarsely here. Logging finer than needed is recoverable; logging
# coarser than the real rate is not, and would mean re-running collection once
# the real contract is measured.


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

EVENNESS_CSV_PATH = os.path.join(RUN_DIR, "sidewalk_evenness_timeseries.csv")
EVENNESS_SUMMARY_PATH = os.path.join(RUN_DIR, "sidewalk_evenness_summary.json")

print(f"Saving walking RGB/depth data to: {RUN_DIR}")
print(f">>> EVENNESS ENABLED: {EVENNESS_ENABLED}")
print(f">>> EVENNESS CSV WILL SAVE TO: {EVENNESS_CSV_PATH}")
print(f">>> EVENNESS SUMMARY WILL SAVE TO: {EVENNESS_SUMMARY_PATH}")


# =====================================================================
# SIDEWALK GEOMETRY
# =====================================================================

# Single source of truth, used for BOTH semantic tagging and evenness
# filtering. Keeping two copies in sync by hand is how the raycast ends up
# measuring a surface the perception stack does not consider sidewalk.
SIDEWALK_PRIMS = [
    # Parametrically generated walkway, imported by
    # scripts/terrain/import_terrain_isaac.py as chunk meshes under this root.
    # Matching is by prefix, so the root covers every chunk. Without this entry
    # every sample on generated terrain is flagged off-sidewalk and silently
    # dropped from the evenness windows.
    "/World/parametric_sidewalk",

    # City scene (AECO + Meshy assets), used as the Tier 2 test set.
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

    # Every walkable surface (the sidewalk mesh plus the defect patches that
    # sit on it) is tagged "sidewalk". These are the same meshes that were
    # walkable before, so the navigation mask is unchanged -- only the label.
    # Curbs and everything else stay untagged and read as "not sidewalk".
    SIDEWALK = SIDEWALK_PRIMS

    tagged = 0

    for p in SIDEWALK:
        prim = stage.GetPrimAtPath(p)

        if prim.IsValid():
            tag(prim, "sidewalk")
            tagged += 1
        else:
            print(f"⚠ Prim not found for semantic tag: {p}")

    print(f"✓ Tagged {tagged} prims as sidewalk")


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

    # Live perception + coverage windows
    "viz_provider": None,
    "viz_window": None,
    "coverage_provider": None,
    "coverage_window": None,
    "last_live_step": 0,

    # Coverage grid
    "coverage_covered": set(),   # cells swept by the sensor swath
    "coverage_track": [],        # spot world (x,y) over time
    "last_coverage_step": 0,

    # Coverage weave
    "sweep_dir": 1.0,            # +1 = aim toward right edge, -1 = left edge
    "sweep_flips": 0,

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

    # Online sidewalk evenness
    "evenness_samples": [],
    "evenness_rows": [],
    "evenness_last_pos": None,
    "evenness_path_distance": 0.0,
    "evenness_prev_body_z": None,
    "evenness_prev_vz": None,
    "evenness_last_class": None,
    "evenness_last_metrics": None,

    # Proprioceptive sensing
    "imu_sensor": None,
    "feet_view": None,           # RigidPrim view over the four feet
    "feet_view_order": [],       # row index -> foot name
    "joint_index": {},           # joint name -> articulation DOF index
    "sensors_ready": False,
    "sensor_warned": set(),      # one-time warnings, so a long run stays quiet

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
    near_center_x = (a + b) / 2.0
    sidewalk_width = b - a + 1

    # ---------------------------------------------------------------
    # Follow the sidewalk DIRECTION: blend the near-band center with the
    # lookahead-band centroid so Spot turns to follow where the walk goes
    # (curves / drift), not just what's at its feet. This is the
    # "assess left/right" signal that was missing.
    # ---------------------------------------------------------------
    look_cols = lookahead.sum(axis=0).astype(np.float64)
    if look_cols.sum() > 0:
        lookahead_center_x = float(
            (np.arange(w) * look_cols).sum() / look_cols.sum()
        )
    else:
        lookahead_center_x = near_center_x

    base_center_x = 0.5 * near_center_x + 0.5 * lookahead_center_x

    centerline_err = (base_center_x - image_center) / image_center
    centerline_err = float(np.clip(centerline_err, -1.0, 1.0))

    # ---------------------------------------------------------------
    # Coverage weave: steer toward one edge of the walk, flipping to the
    # other once this side's near footprint runs out of sidewalk. This
    # makes Spot cover the full width instead of only the centerline.
    # ---------------------------------------------------------------
    if COVERAGE_SWEEP_ENABLED:
        half = sidewalk_width / 2.0

        if _state["sweep_dir"] > 0:
            edge_reached = near_right < SWEEP_EDGE_FRAC * near_total
        else:
            edge_reached = near_left < SWEEP_EDGE_FRAC * near_total

        if edge_reached:
            _state["sweep_dir"] *= -1.0
            _state["sweep_flips"] += 1

        steer_target_x = base_center_x + _state["sweep_dir"] * SWEEP_FRACTION * half
    else:
        steer_target_x = base_center_x

    error = (steer_target_x - image_center) / image_center
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
    # Forward speed scheduling — keyed on CENTERLINE alignment, not the
    # sweep target, so the intentional weave doesn't throttle forward speed.
    # ---------------------------------------------------------------
    center_ratio = near_center / max(near_total, 1)
    abs_c = abs(centerline_err)

    lower_center_weak = center_ratio < 0.15

    if lower_center_weak:
        forward = 0.2

    elif abs_c < 0.20 and sidewalk_width > 0.40 * w:
        forward = NOMINAL_FORWARD_SPEED

    elif abs_c < 0.35:
        forward = 0.6

    elif abs_c < 0.55:
        forward = 0.4

    else:
        forward = 0.2

    # Slow down when turning hard (sharp weave / correction) so Spot stays on
    # the walk, but keep enough speed that the gait keeps striding.
    if abs(turn) > 0.4:
        forward = min(forward, 0.4)

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


def make_sidewalk_binary_image(mask):
    """
    Binary sidewalk / not-sidewalk view from the navigation mask:
    sidewalk pixels green, everything else dark gray, with a small legend.
    """
    if mask is None:
        return None

    mask = np.asarray(mask).astype(bool)
    h, w = mask.shape

    img = np.empty((h, w, 3), dtype=np.uint8)
    img[...] = NOT_SIDEWALK_COLOR
    img[mask] = SIDEWALK_COLOR

    try:
        from PIL import Image as PILImage, ImageDraw

        pil = PILImage.fromarray(img)
        draw = ImageDraw.Draw(pil)
        for i, (label, color) in enumerate(
            [("sidewalk", SIDEWALK_COLOR), ("not sidewalk", NOT_SIDEWALK_COLOR)]
        ):
            y = 4 + i * 16
            draw.rectangle([4, y, 16, y + 12], fill=color, outline=(255, 255, 255))
            draw.text((20, y), label, fill=(255, 255, 255))
        img = np.array(pil)
    except Exception:
        pass

    return img


def draw_region_lines(img):
    """Overlay the navigation region gridlines on a copy of img."""
    img = img.copy()
    h, w = img.shape[:2]

    img[:, w // 3 - 1 : w // 3 + 1] = [255, 0, 0]
    img[:, 2 * w // 3 - 1 : 2 * w // 3 + 1] = [255, 0, 0]

    y1 = h // 2
    y2 = 5 * h // 6
    img[y1 - 1 : y1 + 1, :] = [255, 255, 0]
    img[y2 - 1 : y2 + 1, :] = [255, 255, 0]

    near_y = 2 * h // 3
    img[near_y - 1 : near_y + 1, :] = [0, 255, 255]

    return img


def make_depth_view():
    """
    Colorized depth image from the depth annotator (near = blue, far = red),
    normalized by the 2nd/98th percentiles for contrast.
    """
    depth_data = _state["depth_annotator"].get_data()
    depth = extract_depth_array(depth_data)

    if depth is None:
        return None

    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    lo = np.percentile(depth[valid], 2)
    hi = min(np.percentile(depth[valid], 98), DEPTH_VIEW_MAX_M)
    d = np.clip((depth - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    # Jet-like colormap: near (0) -> blue, far (1) -> red.
    r = np.clip(1.5 - np.abs(4.0 * d - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * d - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * d - 1.0), 0.0, 1.0)

    img = (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)
    img[~valid] = 0
    return img


# =====================================================================
# COVERAGE GRID
# =====================================================================

def _cell_of(x, y):
    return (int(np.floor(x / COVERAGE_CELL_M)),
            int(np.floor(y / COVERAGE_CELL_M)))


def get_spot_world_pose():
    """Return (x, y, heading_rad) of Spot in the world, from the stage."""
    stage = omni.usd.get_context().get_stage()
    body = stage.GetPrimAtPath(f"{SPOT_PRIM_PATH}/body")
    if not body.IsValid():
        return None
    m = UsdGeom.Xformable(body).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    pos = m.ExtractTranslation()
    fwd = m.TransformDir((1.0, 0.0, 0.0))
    heading = float(np.arctan2(fwd[1], fwd[0]))
    return float(pos[0]), float(pos[1]), heading


def mark_coverage_swath():
    """Stamp the sensor swath (a rectangle ahead of Spot) into the covered set."""
    if not COVERAGE_ENABLED:
        return
    pose = get_spot_world_pose()
    if pose is None:
        return
    cx, cy, heading = pose
    _state["coverage_track"].append((cx, cy))

    c, s = np.cos(heading), np.sin(heading)
    half_w = COVERAGE_SWATH_WIDTH_M / 2.0
    step = COVERAGE_CELL_M / 2.0
    us = np.arange(-half_w, half_w + 1e-9, step)
    vs = np.arange(COVERAGE_SWATH_AHEAD_M,
                   COVERAGE_SWATH_AHEAD_M + COVERAGE_SWATH_LEN_M + 1e-9, step)
    covered = _state["coverage_covered"]
    for v in vs:
        for u in us:
            wx = cx + (c * v - s * u)
            wy = cy + (s * v + c * u)
            covered.add(_cell_of(wx, wy))


def render_coverage_map():
    """Top-down RGB coverage image: covered cells green, Spot track blue."""
    covered = _state["coverage_covered"]
    track = _state["coverage_track"]
    if not covered:
        return None

    xs = [c[0] for c in covered]
    ys = [c[1] for c in covered]
    min_i, max_i = min(xs), max(xs)
    min_j, max_j = min(ys), max(ys)
    W = max_i - min_i + 1
    H = max_j - min_j + 1
    if W * H > 20_000_000:
        return None

    img = np.zeros((H, W, 3), dtype=np.uint8)
    for (i, j) in covered:
        img[j - min_j, i - min_i] = (60, 200, 90)   # covered = green

    for (wx, wy) in track:
        i, j = _cell_of(wx, wy)
        ii, jj = i - min_i, j - min_j
        if 0 <= jj < H and 0 <= ii < W:
            img[jj, ii] = (80, 140, 235)             # track = blue

    return np.flipud(img)   # +y world points up


def save_coverage_map():
    """Write the coverage map PNG and print swept-area stats."""
    if not COVERAGE_ENABLED:
        return
    img = render_coverage_map()
    if img is None:
        print("⚠ No coverage cells recorded; skipping map.")
        return

    Image.fromarray(img).save(COVERAGE_MAP_PATH)
    area = len(_state["coverage_covered"]) * COVERAGE_CELL_M * COVERAGE_CELL_M
    print(f"✓ Coverage map saved to {COVERAGE_MAP_PATH}")
    print(f"  covered cells: {len(_state['coverage_covered'])}  "
          f"(~{area:.2f} m^2)  track points: {len(_state['coverage_track'])}")


# =====================================================================
# LIVE PERCEPTION VIEW (omni.ui window inside Isaac Sim)
# =====================================================================

def _make_image_window(title, width, height, px, py):
    """Create an omni.ui window holding a single provider-backed image."""
    import omni.ui as ui

    provider = ui.ByteImageProvider()
    window = ui.Window(
        title, width=width, height=height, position_x=px, position_y=py
    )
    with window.frame:
        try:
            image = ui.ImageWithProvider(provider)
            for attr in ("IWP_PRESERVE_ASPECT_FIT", "PRESERVE_ASPECT_FIT"):
                if hasattr(ui.IwpFillPolicy, attr):
                    image.fill_policy = getattr(ui.IwpFillPolicy, attr)
                    break
        except Exception as inner:
            print(f"  (image widget fallback, no fill_policy: {inner})")
            ui.ImageWithProvider(provider)

    window.visible = True
    try:
        window.focus()
    except Exception:
        pass
    return provider, window


def _push_image(provider, frame_rgb, max_height=None):
    """Push an HxWx3 uint8 RGB image to a ByteImageProvider."""
    if provider is None or frame_rgb is None:
        return

    if max_height and frame_rgb.shape[0] > max_height:
        from PIL import Image as PILImage

        scale = max_height / frame_rgb.shape[0]
        new_size = (int(frame_rgb.shape[1] * scale), max_height)
        frame_rgb = np.array(PILImage.fromarray(frame_rgb).resize(new_size))

    h, w = frame_rgb.shape[:2]
    rgba = np.dstack(
        [frame_rgb, np.full((h, w), 255, dtype=np.uint8)]
    ).astype(np.uint8)
    provider.set_bytes_data(rgba.flatten().tolist(), [w, h])


def setup_live_view():
    """Create (or recreate) the perception + coverage windows."""
    if not LIVE_VIEW_ENABLED:
        return

    try:
        import omni.ui as ui  # noqa: F401  (ensure importable before use)

        print("… setup_live_view: creating omni.ui windows")

        for key in ("viz_window", "coverage_window"):
            old = _state.get(key)
            if old is not None:
                try:
                    old.destroy()
                except Exception:
                    pass
            _state[key] = None
        _state["viz_provider"] = None
        _state["coverage_provider"] = None

        _state["viz_provider"], _state["viz_window"] = _make_image_window(
            "Spot Perception", 1000, 430, 40, 40
        )
        _state["coverage_provider"], _state["coverage_window"] = _make_image_window(
            "Coverage Map", 460, 460, 1060, 40
        )

        print(
            "✓ Live windows created — find them under the top Window menu:\n"
            "   Window > Spot Perception   (RGB | depth | sidewalk/not)\n"
            "   Window > Coverage Map      (green = swept, blue = path)"
        )
    except Exception as e:
        import traceback

        print(f"⚠ Could not create live view windows: {e}")
        traceback.print_exc()
        _state["viz_provider"] = None
        _state["coverage_provider"] = None


def update_live_view(rgb, mask):
    """Push the RGB | depth | sidewalk-binary composite to the window."""
    provider = _state.get("viz_provider")
    if provider is None:
        return

    try:
        rgb_u8 = rgb_to_uint8(rgb)
        if rgb_u8 is None:
            return
        h0 = rgb_u8.shape[0]

        panels = [draw_region_lines(rgb_u8)]

        for panel in (make_depth_view(), make_sidewalk_binary_image(mask)):
            if panel is None:
                continue
            if panel.shape[0] != h0:
                from PIL import Image as PILImage

                new_w = int(panel.shape[1] * h0 / panel.shape[0])
                panel = np.array(
                    PILImage.fromarray(panel).resize((new_w, h0))
                )
            panels.append(panel)

        frame = np.hstack(panels)
        _push_image(provider, frame, max_height=LIVE_VIEW_MAX_HEIGHT)
    except Exception as e:
        print(f"Live view error: {e}")


def update_coverage_view():
    """Push the live top-down coverage map to the Coverage Map window."""
    provider = _state.get("coverage_provider")
    if provider is None:
        return

    try:
        cov = render_coverage_map()
        if cov is None:
            return

        # Nearest-neighbour upscale so early, tiny maps are still visible.
        from PIL import Image as PILImage

        scale = max(1, int(440 / max(cov.shape[0], 1)))
        if scale > 1:
            cov = np.array(
                PILImage.fromarray(cov).resize(
                    (cov.shape[1] * scale, cov.shape[0] * scale),
                    PILImage.NEAREST,
                )
            )
        _push_image(provider, cov, max_height=600)
    except Exception as e:
        print(f"Coverage view error: {e}")


# =====================================================================
# ONLINE SIDEWALK EVENNESS HELPERS
# =====================================================================

def _is_ignored_collider(path, ignored_roots):
    path = str(path or "")
    return any(path == root or path.startswith(root.rstrip("/") + "/")
               for root in ignored_roots)


def _is_sidewalk_collider(path):
    """True if a ray hit belongs to geometry we call sidewalk.

    The Meshy sidewalk sections do not tile continuously -- there are gaps
    between them where a downward ray falls straight through to the AECO
    street mesh underneath. Those hits are perfectly flat road, and mixing
    them into a 1 m evenness window computes peak-to-peak across two
    unrelated objects: a measured 'defect' that is really the edge of the
    sidewalk itself.

    Measured on run_20260914_152758: 610 of 4500 samples (13.6%) landed on
    /World/assembly_City/ce_Context_City_Streets/... with a ground_z range of
    exactly 0.000 m. Real surfaces are never exactly flat; that signature is
    how the contamination was spotted.
    """
    path = str(path or "")
    return any(path == root or path.startswith(root.rstrip("/") + "/")
               for root in SIDEWALK_PRIMS)


def raycast_closest_filtered(origin, direction, max_dist, ignored_roots=()):
    """Return the closest ray hit that is not under an ignored prim root."""
    sq = get_physx_scene_query_interface()
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        return None
    direction = direction / norm

    travelled = 0.0
    epsilon = 1e-4
    while travelled < max_dist:
        remaining = float(max_dist - travelled)
        result = sq.raycast_closest(origin.tolist(), direction.tolist(), remaining)
        if not result or not result.get("hit"):
            return None

        collision_path = str(result.get("collision", ""))
        if not _is_ignored_collider(collision_path, ignored_roots):
            return result

        hit_position = np.asarray(result["position"], dtype=float)
        advance = float(np.dot(hit_position - origin, direction)) + epsilon
        advance = max(advance, epsilon)
        origin = origin + direction * advance
        travelled += advance

    return None


def _body_pose_for_evenness():
    stage = omni.usd.get_context().get_stage()
    body = stage.GetPrimAtPath(f"{SPOT_PRIM_PATH}/body")
    if not body.IsValid():
        return None
    matrix = np.asarray(
        UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),
        dtype=float,
    )
    position = matrix[3, :3].copy()
    forward = matrix[0, :3].copy()
    forward[2] = 0.0
    n = np.linalg.norm(forward)
    if n > 1e-9:
        forward /= n
    return position, forward


def _ground_beneath_body(body_position):
    origin = np.asarray(body_position, dtype=float).copy()
    origin[2] += EVENNESS_RAY_HEIGHT_M
    result = raycast_closest_filtered(
        origin,
        (0.0, 0.0, -1.0),
        EVENNESS_RAY_HEIGHT_M + EVENNESS_RAY_MAX_DIST_M,
        ignored_roots=EVENNESS_IGNORE_ROOTS,
    )
    if not result:
        return None
    hit = np.asarray(result["position"], dtype=float)
    return hit, str(result.get("collision", ""))


def _warn_once(key, message):
    """Print a sensor warning the first time only. A 200 s run samples at
    100 Hz; an unconditional warning would emit 20,000 lines."""
    if key not in _state["sensor_warned"]:
        _state["sensor_warned"].add(key)
        print(f"⚠ {message}")


def create_sensors(world):
    """Create the IMU and per-foot contact sensor PRIMS.

    Must run BEFORE world.reset_async(). Isaac's sensor extension binds
    sensors to the physics sim view at reset; a sensor created after reset is
    never stepped, and reads back a frame full of zeros (IMU) or a permanently
    airborne foot (contact) rather than an error. That failure is silent,
    which is exactly why it is worth a comment.

    Registering with world.scene is what makes the sensor participate in the
    world's own reset/step lifecycle.
    """
    if not SENSORS_ENABLED:
        return

    stage = omni.usd.get_context().get_stage()

    # --- IMU -------------------------------------------------------
    if IMU_ENABLED:
        try:
            from isaacsim.sensors.physics import IMUSensor

            if stage.GetPrimAtPath(IMU_PRIM_PATH):
                stage.RemovePrim(IMU_PRIM_PATH)

            sensor = IMUSensor(
                prim_path=IMU_PRIM_PATH,
                name="proprio_imu",
                frequency=PHYSICS_RATE,
                translation=np.array([0.0, 0.0, 0.0]),
            )
            try:
                world.scene.add(sensor)
            except Exception as exc:
                _warn_once("imu_scene", f"IMU not added to scene: {exc}")
            _state["imu_sensor"] = sensor
            print(f"✓ IMU sensor created at {IMU_PRIM_PATH}")
        except Exception as exc:
            _state["imu_sensor"] = None
            print(f"⚠ IMU sensor unavailable: {exc}")

    # --- foot contact forces ---------------------------------------
    #
    # Net contact forces off a RigidPrim view, NOT isaacsim.sensors.physics
    # ContactSensor. The sensor approach was tried and abandoned: it produced
    # well-formed frames reading exactly zero force forever, even after
    # applying PhysxContactReportAPI to the feet by hand. A sensor sphere has
    # to overlap the contact geometry to register, and that coupling is one
    # more thing to get silently wrong. The view asks PhysX directly for the
    # net force on each foot body and brings its own contact reporting via
    # prepare_contact_sensors.
    if CONTACT_SENSORS_ENABLED:
        try:
            from isaacsim.core.prims import RigidPrim

            # Explicit alternation, so the resolved order is predictable
            # rather than whatever a wildcard happens to enumerate.
            expr = f"{SPOT_PRIM_PATH}/(" + "|".join(FOOT_NAMES) + ")"

            view = RigidPrim(
                prim_paths_expr=expr,
                name="feet_contact_view",
                track_contact_forces=True,
                prepare_contact_sensors=True,
            )
            try:
                world.scene.add(view)
            except Exception as exc:
                _warn_once("feet_view_scene",
                           f"feet view not added to scene: {exc}")
            _state["feet_view"] = view
            print(f"✓ Foot contact view created ({expr})")
        except Exception as exc:
            _state["feet_view"] = None
            print(f"⚠ Foot contact view unavailable: {exc}")


def setup_sensors():
    """Initialize the created sensors and resolve joint indices.

    Runs after play + settle, because the articulation is only initialized on
    the first physics step and its DOF names cannot be read before that.
    Failures are non-fatal: affected fields log NaN and the walk continues,
    since losing a 200 s run to a sensor typo is worse than losing a column.
    """
    if not SENSORS_ENABLED:
        return

    # --- initialize created sensors --------------------------------
    if _state["imu_sensor"] is not None:
        try:
            _state["imu_sensor"].initialize()
            print("✓ IMU sensor initialized")
        except Exception as exc:
            print(f"⚠ IMU initialize failed: {exc}")

    view = _state["feet_view"]
    if view is not None:
        try:
            try:
                view.initialize()
            except Exception:
                pass  # already initialized by world.reset()

            # Resolve which row of the force array belongs to which foot.
            # Never assume the view enumerates in FOOT_NAMES order.
            paths = list(getattr(view, "prim_paths", []) or [])
            order = []
            for path in paths:
                leaf = str(path).rstrip("/").split("/")[-1]
                order.append(leaf if leaf in FOOT_NAMES else None)
            _state["feet_view_order"] = order

            if any(f is None for f in order) or len(order) != len(FOOT_NAMES):
                print(f"⚠ foot view order unexpected: {paths}")
            else:
                print(f"✓ Foot contact view initialized, order: {order}")
        except Exception as exc:
            _state["feet_view"] = None
            print(f"⚠ Foot contact view initialize failed: {exc}")

    # --- joint index map -------------------------------------------
    # Resolved by NAME. USD traversal order is not articulation DOF order, and
    # indexing joint arrays positionally would silently mislabel every column.
    try:
        robot = getattr(_state["spot"], "robot", None)
        dof_names = list(getattr(robot, "dof_names", []) or [])
        if dof_names:
            _state["joint_index"] = {
                name: dof_names.index(name)
                for name in JOINT_NAMES if name in dof_names
            }
            missing = [n for n in JOINT_NAMES if n not in dof_names]
            if missing:
                print(f"⚠ joints not in articulation: {missing}")
            print(f"✓ Resolved {len(_state['joint_index'])}/{len(JOINT_NAMES)} "
                  f"joint indices from dof_names")
        else:
            print("⚠ articulation exposed no dof_names; joint logging disabled")
    except Exception as exc:
        print(f"⚠ Joint index resolution failed: {exc}")

    _state["sensors_ready"] = True


def _read_imu():
    """Body IMU: linear acceleration and angular velocity.

    Isaac's IMU reports proper acceleration in the sensor frame, so it
    includes the gravity reaction the way a physical accelerometer does --
    a stationary level robot reads about +9.81 on Z, not 0. Left as measured
    rather than gravity-compensated, because that is what the real device
    hands over and the feature extractor should see the same thing in both.
    """
    sensor = _state["imu_sensor"]
    if sensor is None:
        return {}
    try:
        frame = sensor.get_current_frame()
        lin = np.asarray(frame.get("lin_acc", [np.nan] * 3), dtype=float)
        ang = np.asarray(frame.get("ang_vel", [np.nan] * 3), dtype=float)
        return {
            "imu_lin_acc_x": float(lin[0]),
            "imu_lin_acc_y": float(lin[1]),
            "imu_lin_acc_z": float(lin[2]),
            "imu_ang_vel_x": float(ang[0]),
            "imu_ang_vel_y": float(ang[1]),
            "imu_ang_vel_z": float(ang[2]),
        }
    except Exception as exc:
        _warn_once("imu_read", f"IMU read failed: {exc}")
        return {}


def _read_feet(body_position, body_forward):
    """Per-foot contact state and position relative to the body.

    Mirrors the real SDK's foot_state[]: contact flag, contact force, and
    foot position in a body-relative frame. Touchdown height is the signal
    most likely to survive sim-to-real, because where a foot comes to rest is
    geometry rather than controller behaviour.
    """
    out = {}
    stage = omni.usd.get_context().get_stage()

    # Net contact force per foot, fetched once for all four.
    forces_by_foot = {}
    view = _state["feet_view"]
    if view is not None:
        try:
            # dt matters: PhysX accumulates a contact IMPULSE over the step and
            # this call divides by dt to give a force. The default dt=1.0 would
            # report the raw impulse -- physically an N*s, labelled newtons, and
            # 500x too small at a 500 Hz step. Passing the real physics dt is
            # what turns ~0.6 into the ~300 N a 25 kg robot actually carries.
            forces = view.get_net_contact_forces(dt=1.0 / PHYSICS_RATE)
            if forces is not None:
                forces = np.asarray(forces, dtype=float).reshape(-1, 3)
                order = _state["feet_view_order"]
                for row, foot in enumerate(order):
                    if foot is not None and row < forces.shape[0]:
                        forces_by_foot[foot] = float(
                            np.linalg.norm(forces[row]))
        except Exception as exc:
            _warn_once("feet_forces", f"net contact force read failed: {exc}")

    # Body frame: forward (x), left (y), up (z).
    fwd = np.asarray(body_forward, dtype=float)
    fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    left = np.cross(up, fwd)
    left = left / max(np.linalg.norm(left), 1e-9)

    for foot in FOOT_NAMES:
        prefix = f"foot_{foot}"

        # Position, from the USD transform.
        try:
            prim = stage.GetPrimAtPath(f"{SPOT_PRIM_PATH}/{foot}")
            matrix = np.asarray(
                UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()),
                dtype=float,
            )
            world = matrix[3, :3]
            rel = world - np.asarray(body_position, dtype=float)
            out[f"{prefix}_x_m"] = float(np.dot(rel, fwd))
            out[f"{prefix}_y_m"] = float(np.dot(rel, left))
            out[f"{prefix}_z_m"] = float(rel[2])
            out[f"{prefix}_world_z_m"] = float(world[2])
        except Exception as exc:
            _warn_once(f"{prefix}_pose", f"{foot} pose read failed: {exc}")
            for suffix in ("x_m", "y_m", "z_m", "world_z_m"):
                out[f"{prefix}_{suffix}"] = np.nan

        # Contact, from the net force magnitude on the foot body.
        force = forces_by_foot.get(foot, np.nan)
        out[f"{prefix}_force_n"] = force
        out[f"{prefix}_in_contact"] = (
            np.nan if not np.isfinite(force)
            else int(force >= CONTACT_FORCE_THRESHOLD_N)
        )

    return out


def _read_joints():
    """Joint positions, velocities and measured efforts, by name.

    Effort is the most controller-dependent signal logged here -- it is the
    control action taken to reject the disturbance, and the real robot's
    controller is not this one. Logged anyway so the modality ablation can
    measure how much it actually helps, rather than assuming.
    """
    index = _state["joint_index"]
    if not index:
        return {}

    robot = getattr(_state["spot"], "robot", None)
    if robot is None:
        return {}

    def _fetch(getter_names):
        for name in getter_names:
            getter = getattr(robot, name, None)
            if getter is None:
                continue
            try:
                value = getter()
                if value is not None:
                    return np.asarray(value, dtype=float).ravel()
            except Exception:
                continue
        return None

    positions = _fetch(["get_joint_positions"])
    velocities = _fetch(["get_joint_velocities"])
    efforts = _fetch(["get_measured_joint_efforts", "get_applied_joint_efforts"])

    out = {}
    for name, i in index.items():
        for label, array in (("pos_rad", positions),
                             ("vel_rads", velocities),
                             ("effort_nm", efforts)):
            if array is not None and i < array.size:
                out[f"joint_{name}_{label}"] = float(array[i])
            else:
                out[f"joint_{name}_{label}"] = np.nan
    return out


def _detrended_height_metrics(samples):
    """Compute local surface relief after removing best-fit grade."""
    if len(samples) < EVENNESS_MIN_SAMPLES:
        return None

    distance = np.asarray([s["distance_m"] for s in samples], dtype=float)
    height = np.asarray([s["ground_z_m"] for s in samples], dtype=float)
    span = float(distance[-1] - distance[0])
    if span < EVENNESS_MIN_WINDOW_M:
        return None

    coeff = np.polyfit(distance, height, 1)
    trend = np.polyval(coeff, distance)
    residual = height - trend

    ptp_mm = float(np.ptp(residual) * 1000.0)
    rms_mm = float(np.sqrt(np.mean(residual ** 2)) * 1000.0)
    max_abs_mm = float(np.max(np.abs(residual)) * 1000.0)

    accel = np.asarray([s["body_vertical_accel_mps2"] for s in samples], dtype=float)
    accel = accel[np.isfinite(accel)]
    accel_rms = float(np.sqrt(np.mean(accel ** 2))) if accel.size else None
    accel_peak = float(np.max(np.abs(accel))) if accel.size else None

    if ptp_mm <= EVENNESS_SMOOTH_MAX_MM:
        label = "smooth"
    elif ptp_mm <= EVENNESS_MILD_MAX_MM:
        label = "mild"
    elif ptp_mm <= EVENNESS_UNEVEN_MAX_MM:
        label = "uneven"
    else:
        label = "severe"

    return {
        "window_start_m": float(distance[0]),
        "window_end_m": float(distance[-1]),
        "window_span_m": span,
        "grade_percent": float(coeff[0] * 100.0),
        "height_rms_mm": rms_mm,
        "height_peak_to_peak_mm": ptp_mm,
        "height_max_abs_mm": max_abs_mm,
        "body_vertical_accel_rms_mps2": accel_rms,
        "body_vertical_accel_peak_mps2": accel_peak,
        "classification": label,
    }


def sample_sidewalk_evenness(step_size):
    """Collect one synchronized terrain/body sample and update online metrics."""
    if not EVENNESS_ENABLED:
        return

    sc = _state["step_count"]
    if sc < SETTLE_STEPS or sc % EVENNESS_SAMPLE_EVERY_N_STEPS != 0:
        return

    pose = _body_pose_for_evenness()
    if pose is None:
        return
    body_position, body_forward = pose

    ground = _ground_beneath_body(body_position)
    if ground is None:
        return
    hit_position, collider = ground
    on_sidewalk = _is_sidewalk_collider(collider)

    previous_position = _state["evenness_last_pos"]
    if previous_position is not None:
        delta_xy = body_position[:2] - previous_position[:2]
        _state["evenness_path_distance"] += float(np.linalg.norm(delta_xy))
    _state["evenness_last_pos"] = body_position.copy()

    sample_dt = float(step_size) * EVENNESS_SAMPLE_EVERY_N_STEPS
    previous_z = _state["evenness_prev_body_z"]
    previous_vz = _state["evenness_prev_vz"]
    vertical_accel = np.nan
    if previous_z is not None and sample_dt > 0:
        vz = float((body_position[2] - previous_z) / sample_dt)
        if previous_vz is not None:
            vertical_accel = float((vz - previous_vz) / sample_dt)
        _state["evenness_prev_vz"] = vz
    _state["evenness_prev_body_z"] = float(body_position[2])

    row = {
        "step": int(sc),
        "time_s": float(sc / PHYSICS_RATE),
        "distance_m": float(_state["evenness_path_distance"]),
        "body_x_m": float(body_position[0]),
        "body_y_m": float(body_position[1]),
        "body_z_m": float(body_position[2]),
        # Horizontal heading, as a unit vector rather than an angle -- no
        # wrap-around, no convention to get wrong. Needed to put a
        # body-relative foot position back into a fixed frame, which is what
        # proprioceptive terrain reconstruction requires. The real SDK gives
        # the equivalent via body pose in the odom frame.
        "body_fwd_x": float(body_forward[0]),
        "body_fwd_y": float(body_forward[1]),
        "ground_x_m": float(hit_position[0]),
        "ground_y_m": float(hit_position[1]),
        "ground_z_m": float(hit_position[2]),
        "body_clearance_m": float(body_position[2] - hit_position[2]),
        # Kept for continuity with earlier runs, but NOT a sensor: this is body
        # Z double-differenced from the USD transform. It has none of a real
        # IMU's error characteristics, and it is derived from the same body
        # pose the labels are anchored to. Train on imu_lin_acc_*, not this.
        "body_vertical_accel_mps2": vertical_accel,
        "collider": collider,
        # 1 when the ray landed on sidewalk geometry, 0 when it fell through a
        # gap onto the city street. Off-sidewalk rows are still logged -- the
        # proprioception is valid regardless of what is underfoot -- but they
        # are kept out of the rolling window below so a classification never
        # spans two different surfaces.
        "on_sidewalk": int(on_sidewalk),
    }

    # Proprioception -- what a real Spot could actually feel. Synchronised to
    # the label by construction, since it lands in the same row.
    if SENSORS_ENABLED and _state["sensors_ready"]:
        try:
            row.update(_read_imu())
            row.update(_read_feet(body_position, body_forward))
            row.update(_read_joints())
        except Exception as exc:
            _warn_once("sensor_row", f"sensor sampling failed: {exc}")

    _state["evenness_rows"].append(row)
    if on_sidewalk:
        _state["evenness_samples"].append(row)

    # Periodically persist the raw CSV so the file exists during the run and
    # survives an interrupted Script Editor task. Rewriting is acceptable here
    # because runs are modest in size and it avoids partial-row corruption.
    if len(_state["evenness_rows"]) % EVENNESS_AUTOSAVE_EVERY_N_SAMPLES == 0:
        _write_evenness_csv()
        print(
            f">>> EVENNESS AUTOSAVE: {len(_state['evenness_rows'])} samples -> "
            f"{EVENNESS_CSV_PATH}"
        )

    # Keep only the most recent distance window.
    cutoff = _state["evenness_path_distance"] - EVENNESS_WINDOW_M
    _state["evenness_samples"] = [
        sample for sample in _state["evenness_samples"]
        if sample["distance_m"] >= cutoff
    ]

    metrics = _detrended_height_metrics(_state["evenness_samples"])
    if metrics is None:
        return

    _state["evenness_last_metrics"] = metrics
    label = metrics["classification"]
    changed = label != _state["evenness_last_class"]
    periodic = sc % PHYSICS_RATE == 0
    if changed or periodic:
        print(
            f">>> SIDEWALK {label.upper():7s}  "
            f"d={metrics['window_end_m']:.2f}m  "
            f"ptp={metrics['height_peak_to_peak_mm']:.1f}mm  "
            f"rms={metrics['height_rms_mm']:.1f}mm  "
            f"grade={metrics['grade_percent']:+.1f}%  "
            f"az_rms={metrics['body_vertical_accel_rms_mps2'] or 0.0:.2f}m/s²"
        )
    _state["evenness_last_class"] = label


def _write_evenness_csv():
    """Write all raw evenness rows to disk atomically enough for inspection."""
    rows = _state["evenness_rows"]
    if not rows:
        return False

    # Column set is the UNION across all rows, in first-seen order. Taking it
    # from row 0 alone would raise as soon as sensors come online partway
    # through a run, or drop every sensor column if row 0 predates them.
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(EVENNESS_CSV_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return True


def save_evenness_results():
    """Write raw time series plus a run-level summary."""
    if not EVENNESS_ENABLED or not _state["evenness_rows"]:
        print("⚠ No sidewalk evenness samples recorded.")
        print(f"  expected CSV path: {EVENNESS_CSV_PATH}")
        return

    _write_evenness_csv()

    # Summary metrics use only on-sidewalk samples, for the same reason the
    # live rolling window does: a window spanning a gap measures the step off
    # the sidewalk, not a defect in it. The raw CSV keeps every row.
    all_rows = _state["evenness_rows"]
    rows = [r for r in all_rows if r.get("on_sidewalk", 1)]
    off = len(all_rows) - len(rows)
    if off:
        print(f"  excluded {off}/{len(all_rows)} samples "
              f"({off / len(all_rows) * 100:.1f}%) that hit non-sidewalk geometry")
    if not rows:
        print("⚠ No on-sidewalk samples; summary skipped.")
        return

    all_metrics = _detrended_height_metrics(rows)
    labels = []
    # Recompute classifications across 1 m windows at approximately 10 cm spacing.
    next_end = EVENNESS_MIN_WINDOW_M
    for i, row in enumerate(rows):
        if row["distance_m"] < next_end:
            continue
        start_d = row["distance_m"] - EVENNESS_WINDOW_M
        window = [r for r in rows[:i + 1] if r["distance_m"] >= start_d]
        metric = _detrended_height_metrics(window)
        if metric:
            labels.append(metric)
        next_end = row["distance_m"] + 0.10

    summary = {
        "run_name": RUN_NAME,
        "sample_count": len(rows),
        "path_length_m": float(rows[-1]["distance_m"]),
        "thresholds_mm": {
            "smooth_max": EVENNESS_SMOOTH_MAX_MM,
            "mild_max": EVENNESS_MILD_MAX_MM,
            "uneven_max": EVENNESS_UNEVEN_MAX_MM,
        },
        "whole_run_metrics": all_metrics,
        "window_metrics": labels,
    }
    with open(EVENNESS_SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"✓ Sidewalk evenness CSV: {EVENNESS_CSV_PATH}")
    print(f"✓ Sidewalk evenness summary: {EVENNESS_SUMMARY_PATH}")
    if all_metrics:
        print(
            f"  whole run: {all_metrics['classification']}  "
            f"ptp={all_metrics['height_peak_to_peak_mm']:.1f} mm  "
            f"rms={all_metrics['height_rms_mm']:.1f} mm"
        )


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

    try:
        sample_sidewalk_evenness(step_size)
    except Exception as exc:
        if sc % PHYSICS_RATE == 0:
            print(f"Evenness sampling error: {exc}")

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
        _state["last_live_step"] = 0
        _state["coverage_covered"] = set()
        _state["coverage_track"] = []
        _state["last_coverage_step"] = 0
        _state["sweep_dir"] = 1.0
        _state["sweep_flips"] = 0
        _state["recovery_mode"] = None
        _state["recovery_reason"] = ""
        _state["recovery_cycles"] = 0
        _state["evenness_samples"] = []
        _state["evenness_rows"] = []
        _state["evenness_last_pos"] = None
        _state["evenness_path_distance"] = 0.0
        _state["evenness_prev_body_z"] = None
        _state["evenness_prev_vz"] = None
        _state["evenness_last_class"] = None
        _state["evenness_last_metrics"] = None
        _state["imu_sensor"] = None
        _state["feet_view"] = None
        _state["feet_view_order"] = []
        _state["joint_index"] = {}
        _state["sensors_ready"] = False
        _state["sensor_warned"] = set()
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

        # Sensors must exist and be registered BEFORE reset -- Isaac binds them
        # to the physics sim view there. Created after reset they are never
        # stepped and read back zeros instead of failing.
        create_sensors(world)

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

        setup_live_view()

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

        # After play + settle, so the articulation has been initialized by the
        # first physics step and its dof_names can be read.
        setup_sensors()

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

                    # Live perception window (throttled)
                    if (
                        LIVE_VIEW_ENABLED
                        and sc - _state["last_live_step"] >= LIVE_VIEW_EVERY_N_STEPS
                    ):
                        _state["last_live_step"] = sc
                        update_live_view(rgb_data, mask)

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
                            f"sweep_dir={_state['sweep_dir']:+.0f}  "
                            f"sweep_flips={_state['sweep_flips']}  "
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

            # Coverage: stamp the swath and refresh the live coverage map
            if (
                COVERAGE_ENABLED
                and sc >= SETTLE_STEPS
                and sc - _state["last_coverage_step"] >= COVERAGE_MARK_EVERY_N_STEPS
            ):
                _state["last_coverage_step"] = sc

                try:
                    mark_coverage_swath()
                    update_coverage_view()
                except Exception as e:
                    print(f"Coverage error: {e}")

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

        save_coverage_map()
        save_evenness_results()
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