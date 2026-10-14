# Sim-to-real sidewalk unevenness detection — plan

**Goal.** Train a model entirely in simulation that detects sidewalk unevenness
from Spot's onboard proprioceptive sensing, and have it work on a real Spot
walking a real sidewalk.

**Why this framing.** Simulation gives exact, dense, free labels for a quantity
that is expensive to measure in the real world. Vision-based defect detection
already exists in this repo; the open question is whether the robot can feel
unevenness it cannot see, which is the part that generalizes across lighting,
weather, and surface appearance.

---

## Pivot, 2026-09-14: measurement instead of inference

The goal is an **inspection record** — "at 23.4 m the vertical discontinuity is
14 mm, above the 13 mm ADA barrier threshold." That is a measurement with a
stated accuracy. A learned model gives an estimate with an error distribution,
which is weaker evidence for compliance and needs retraining for every new
robot, gait and surface.

A foot in contact **is** on the surface, so each stance is a direct terrain
sample. No model required. `scripts/model/reconstruct_terrain.py` measures the
profile from footfalls and scores it against ADA thresholds.

**Validated against authored truth**, 32 m walk, 5,728 mid-stance samples,
with body heading logged exactly:

| quantity | result |
|---|---|
| Profile residual | **std 0.88 mm**, p90 0.39 mm |
| Vertical discontinuity | bias +0.14 mm, **MAE 0.42 mm**, corr **+0.999** |
| Roughness (detrended ptp) | bias −1.38 mm, MAE 1.48 mm, corr +0.969 |
| ADA class agreement | **54/58 (93%)** |

Against ADA thresholds of 6.4 mm and 13 mm, sub-millimetre accuracy on the
regulated quantity is comfortably inspection-grade.

**Heading must be logged, not inferred.** An earlier run without `body_fwd_*`
fell back to direction of travel, which differs from body heading whenever the
robot crabs — and the walker's coverage weave makes it crab constantly. That
one approximation cost 3x on discontinuity (MAE 1.33 → 0.42 mm) and 40% on
profile residual (1.45 → 0.88 mm). Every foot position depends on it.

**Cracks read shallower than authored, correctly.** Where the walkway has
narrow cracks the measured roughness sits below truth, because a foot bridges
a 2-6 cm crack rather than entering it. The authored profile includes the
crack floor; the robot measures the walking surface. For ADA purposes the
walking surface is the right answer — the regulated quantity is the change in
level a pedestrian's foot meets, not the depth of a void.

**What made the difference** (a first per-track attempt gave MAE 7.8–11.4 mm
and 57–61% class agreement):

1. **All four feet combined**, not left/right tracks separately — double the
   sample density, and density is what resolves a step.
2. **Plane fit in (x, y), not a line in x.** Left and right feet sit ~0.3 m
   apart, so on a 2% cross slope they differ by ~6 mm for reasons unrelated to
   roughness. Removing the plane lets all four contribute to one profile.
3. **Resampling to a regular 5 cm profile**, so the generator's own estimators
   apply unchanged and measured numbers mean exactly what authored ones mean.

I had called the first attempt's step under-measurement a fundamental limit of
sparse footfall sampling. That was wrong — it was an artefact of applying the
estimator to irregularly spaced per-track samples. Bias went from −10 mm to
−0.18 mm without collecting anything new.

**The simulator's role changes**: no longer a source of training data, it is
the **calibration rig**. Because the terrain was authored, accuracy can be
stated exactly — a claim unavailable on real sidewalks without a survey crew.
Everything built still applies: generator, exact labels, Isaac import at
0.000 mm collision fidelity, sensor logging, the walker. Only the final step
changed.

The ML path (Phase 4) is now optional — an interesting research question about
whether unevenness is detectable from proprioception *without* reconstructing
geometry, not a requirement for the inspection tool.

## Locked decisions

These were settled during design; recorded here so they don't get relitigated.

| Decision | Rationale |
|---|---|
| Training labels come from simulation only | Free, exact, dense. The entire value proposition of the sim approach. |
| Terrain is **parametrically generated**, not asset-placed | The 17 defect GLBs in `sidewalks/` are a 17-shape label distribution. Yaw and ±10% scale jitter re-present the same height profiles; they do not create new roughness. A sim-only model would memorize 17 shapes. |
| Ground-truth labels are **derived from generative parameters**, not measured | If we synthesize an 8.2 mm joint fault, the label is 8.2 mm exactly. No raycast, no plane fit, no confidence gating. |
| The existing raycast profiler stays, as a **check** | `spot_walk_with_evenness_autosave.py` measures the PhysX collision surface. Agreement between intent and physics is a validation loop, not a label source. |
| Thresholds align to **ADA / PROWAG**, not round numbers | 6.4 mm (1/4 in) requires beveling; 13 mm (1/2 in) is a barrier. External standard, directly actionable, and puts the easiest class boundary above the Meshy mesh-artifact scale (~3 mm). |
| **Vertical discontinuity** and **roughness** are separate labels | A 13 mm step and a 13 mm smooth dome are different hazards with different detectability. Collapsing them into one "unevenness" scalar destroys the distinction that matters for compliance. |
| Simulate only what a real Spot can report, at the real rate | A model trained on 500 Hz raw IMU cannot run on a robot that streams fused state at ~50 Hz. The real data contract constrains the sim, not the reverse. |
| Prefer **controller-invariant** features | Sim runs `SpotFlatTerrainPolicy`; the real robot runs Boston Dynamics' controller. Foot touchdown geometry transfers; joint torque signatures encode controller response and will not. |

---

## Phase 1 — Parametric terrain generation `[COMPLETE]`

Replace asset placement with a heightfield synthesizer whose parameters are the
ground truth.

**Deliverables**
- `scripts/terrain/sidewalk_terrain.py` — synthesis + label computation, pure
  numpy/scipy, no Isaac dependency.
- `scripts/terrain/generate_dataset.py` — batch CLI writing heightfields,
  labels, previews and meshes.
- `scripts/terrain/import_terrain_isaac.py` — Isaac-side OBJ import with
  collision, run in the Script Editor.

**Modelled failure modes** (chosen to match how concrete sidewalks actually
fail, not generic rough terrain):

- **Slab joint faulting** — differential settlement between adjacent panels.
  The dominant real trip hazard and the one ADA regulates directly.
- **Tree root heave** — smooth dome uplift, 0.5–2 m wavelength.
- **Settlement depressions** — broad low spots.
- **Slab curl** — panel edges lifting relative to centers as the top surface
  dries faster than the base.
- **Cracks** — narrow channels.
- **Spalling / surface texture** — high-frequency, low-amplitude roughness.
- **Running grade and cross slope** — ADA regulates these (5% / 2%), and the
  detrending step must remove them correctly rather than reading them as
  roughness.

**Labels emitted per surface**
- The generative parameters and every concrete feature placed.
- Per-station, per-lane rolling-window metrics matching the existing evenness
  definition: detrended peak-to-peak and RMS over a 1 m window.
- Maximum vertical discontinuity in the window (the ADA metric).
- ADA class: `compliant` / `bevel_required` / `barrier`.

**Exit criterion.** Generated surfaces span the full severity range with
controllable class balance, and a raycast pass in Isaac over an imported mesh
recovers the authored heights to within a millimetre.

**Status: met, 2026-09-14.** Severity maps monotonically onto discontinuity,
and a 20-surface batch lands at 65% / 23% / 13% across the three ADA classes.
Isaac verification on `sw_00003` (severe, 11 features, 31 barrier windows):
400 rays, 0 missed, **max |error| 0.000 mm** against a 1 mm tolerance.

Rays sample at grid points, which are also mesh vertices, so this confirms the
mesh and collider are exact rather than probing between-vertex interpolation —
which is the right test, since labels are computed on the same 1 cm grid.

Three Isaac-side bugs were fixed to get there, all worth remembering:

- `import omni.timeline` inside a function binds `omni` as a local name for
  that whole function, shadowing the module-level `import omni.usd`. Keep
  submodule imports at module scope.
- PhysX does not see a collider in the frame the USD prim is authored. The
  mesh must be cooked and the scene repopulated first, so verification has to
  tick the app before querying. Raycasting immediately returns zero hits,
  which looks identical to a broken collider.
- `PhysxSDFMeshCollisionAPI` is for dynamic, roughly convex bodies. On a
  static ~400k-triangle heightfield it is very expensive to cook and can leave
  the collider unusable. A static mesh with approximation `none` is already
  exact triangle collision.

Two labelling bugs were found and fixed during the build; both are recorded in
the self-test so they cannot regress:

1. **Step halving.** A moving average applied before differencing smeared each
   step across its own kernel and reported `step / k` — a 12 mm fault read as
   6 mm. The two comparison windows must not straddle the boundary.
2. **Tooled joints read as hazards.** Every scored control joint was being
   labelled a 6–8 mm vertical discontinuity, because the groove wall *is* an
   abrupt height change. But a groove is a void, not a change in level between
   walking surfaces, and ADA regulates the latter. Uncaught, this would have
   taught the model that every normal sidewalk joint is a trip hazard. Narrow
   notches are now filled by a morphological closing before measurement; a
   step is invariant under closing, so genuine faults survive at full height.

The second bug was invisible in the summary statistics and obvious in the
preview image — reason enough to keep rendering previews.

---

## Phase 2 — Sensor logging matched to the real data contract

**Partially answered from the proto, 2026-09-14.** Checked
`bosdyn/api/robot_state.proto` directly rather than waiting for hardware.
**58 of our 62 logged channels have a real equivalent:**

| our channels | SDK field | status |
|---|---|---|
| 36 joint pos/vel/effort | `JointState.position` / `.velocity` / `.load` (N·m) | available |
| 12 foot body-relative xyz | `FootState.foot_position_rt_body` | available |
| 4 foot contact flags | `FootState.contact` (CONTACT_MADE/LOST) | available |
| 6 IMU accel + gyro | `ImuState.Packet.acceleration_rt_odom_in_link_frame`, `.angular_velocity_rt_odom_in_link_frame` | available, packetized high-rate |
| **4 foot contact forces** | **no such field** | **sim-only** |

`FootState` carries contact state and a `TerrainState` (friction estimate,
slip distance/velocity, ground contact normal, visual ground penetration) but
no force magnitude. `baseline.py` now tags these as `SIM_ONLY_CHANNELS` and
reports a `DEPLOYABLE (real-available)` row in every ablation, so a model that
depends on undeployable inputs cannot go unnoticed.

Also available and not yet logged: `JointState.acceleration`, and the whole
`TerrainState` block — worth mirroring in sim later.

**Open convention question.** Our sim IMU reports proper acceleration (9.81 on
Z at rest, gravity included). Spot's field name implies acceleration relative
to the odom frame, i.e. gravity-compensated (~0 at rest). If so the two differ
by gravity rotated into body frame. Convertible — the packet ships
`odom_rot_link` — but it must be done deliberately. Verify on first hardware
contact.

**Still blocked on hardware:** actual streaming rate, real sensor noise
characteristics, and confirmation of the gravity convention.

Expected to be available from `RobotState`, to be confirmed against the SDK
version in use:

- `foot_state[]` — per-foot position relative to body, contact state, and
  (newer SDKs) a per-foot terrain state including ground contact normal, slip,
  and ground penetration estimates.
- `kinematic_state.joint_states[]` — position, velocity, acceleration, load.
- Body pose and velocity from the state estimator.

Notes:
- Spot's body IMU is **fused**, not raw. If raw inertial data is wanted, mount a
  dedicated IMU payload — a signal we fully characterize is a signal we can
  honestly simulate.
- Expect ~30–50 Hz, not 500 Hz. Downsample sim to match **before** windowing.
- The current `body_vertical_accel_mps2` field in the evenness sampler is body
  Z double-differenced from the USD transform. It is not an IMU, it has none of
  the error characteristics of one, and it is derived from the same body pose
  the labels are anchored to. It should be replaced, not extended.

Sim-side work: add the mirrored fields to the existing evenness sample row in
`spot_walk_with_evenness_autosave.py` — same schema, same units, so one
feature-extraction path serves both sim and real.

**Sim half: built 2026-09-14.** Names confirmed against the actual asset by
`scripts/Isaac_sim/probe_spot_state.py` (Isaac Sim 5.1): feet are
`fl_foot`/`fr_foot`/`hl_foot`/`hr_foot`, joints are `{fl,fr,hl,hr}_{hx,hy,kn}`,
and `IMUSensor` / `ContactSensor` / `SingleArticulation` are all available. No
sensors or contact reporters existed on the asset; both are created at setup.

66 columns added to each evenness row — 6 IMU, 24 foot (4 x position,
body-relative position, contact flag, contact force), 36 joint (12 x position,
velocity, effort).

Decisions worth keeping:

- **Joint indices resolved by name from `dof_names` at runtime.** USD traversal
  order is not articulation DOF order; indexing positionally would silently
  mislabel every joint column.
- **Sampling stays at 100 Hz, not downsampled to the expected real rate.**
  Matching the real rate is done by decimating at training time. Logging finer
  than needed is recoverable; logging coarser than the real robot would force
  a re-run of collection once the real contract is measured.
- **IMU left gravity-inclusive**, as a physical accelerometer reports it,
  rather than gravity-compensated — that is what the real device hands over.
- **Sensor failures are non-fatal.** Affected fields log NaN and the walk
  continues; losing a 200 s run to a sensor typo is worse than losing a column.
  Warnings fire once, not 20,000 times.
- **`body_vertical_accel_mps2` retained but marked not-a-sensor** in the code,
  so old runs stay readable while nothing new trains on it.

**Verified working 2026-09-14** over a 1000-sample walk:

| signal | evidence it is real |
|---|---|
| IMU | `imu_lin_acc_z` mean 9.806 m/s² — gravity, as a physical accelerometer reports it. Yaw rate std matches the commanded turn. |
| Foot position | Body-relative stance matches Spot's geometry (front +0.33/+0.40 m, hind −0.31/−0.37 m, lateral ±0.15 m, −0.46 m below body). Foot Z swing 0.10–0.14 m is the gait step height. |
| Foot contact | 50–53% duty cycle per foot; **exactly 2 feet down in 459 of 500 samples** — a trot, which nothing in the code knows about. Total force mean 301 N against ~245 N static, correctly higher under dynamic load. |
| Joints | `fl_kn` −1.603 ± 0.082 rad, effort +16.9 ± 17.8 Nm, velocity std ~2 rad/s. |

Four bugs were found getting here, and **every one failed silently** — no
exception, no error, just well-formed wrong data:

1. **Sensors created after `world.reset_async()`** are never bound to the
   physics sim view. The IMU returned a valid frame of zeros; the feet read as
   permanently airborne. Create and register with `world.scene` BEFORE reset;
   initialize after play.
2. **`ContactSensor` never registered contacts at all**, even with
   `PhysxContactReportAPI` applied to the feet by hand — its sensor sphere has
   to overlap contact geometry. Abandoned for a `RigidPrim` view with
   `track_contact_forces=True`, which asks PhysX for net force on the body
   directly and sets up reporting itself.
3. **`get_net_contact_forces()` defaults to `dt=1.0`.** PhysX accumulates an
   impulse per step and divides by `dt`, so the default reports an impulse
   labelled as newtons — 500x too small at a 500 Hz step. Pass the real
   physics dt.
4. **Contact force lives under the `value` key**, not `force`.

Lesson for the batch runner: add cheap physical assertions to every run —
gravity ≈ 9.81 on a stationary robot, feet alternating contact, total contact
force ≈ body weight, joint efforts nonzero. Catching this in run 1 of 500 is a
very different day from catching it in run 500.

---

## Phase 2.5 — Walkway concatenation `[COMPLETE 2026-09-14]`

A real sidewalk is one continuous walkway made of many slabs, so the robot
should meet it that way: a single long strip walked once, not N short strips
with a stage rebuild between each. `concatenate_surfaces()` joins generated
surfaces end to end; `--segments N` in `generate_dataset.py` produces one.

Verified: 20 segments, **234.5 m, 2336 labelled windows**, class balance
68% / 21% / 11%, seams invisible in the centreline profile.

**Seam handling.** Each surface is re-levelled (its own plane removed) and the
join is cross-faded over 0.3 m. Re-levelling costs nothing because the labels
are grade-invariant by construction — `_detrended_metrics` removes a linear
fit, and cross slope shifts a lane by a constant — while a discontinuous grade
at every seam would be unphysical. Windows straddling a join are flagged
`spans_seam` (~7%) and carry a `segment_id`, which is the train/test split key.

**Three wrong measurements before one right one.** The seam check was rebuilt
three times, and each wrong version made concatenation look broken when it was
not:

1. Differencing strip means either side of the join — reads walkway *grade* as
   a step (2 mm on a 2% grade over a 0.1 m separation).
2. Maximising the discontinuity estimator over a ±0.5 m window — slabs sit
   ~1.35 m apart and faults reach 30 mm, so the window usually contained a
   genuine authored fault and reported it as a seam artefact.
3. Evaluating the estimator exactly at the seam index — correct.

**The lesson is the control, not the estimator.** Absolute numbers were
uninterpretable; what settled it was running the same estimator at random
non-seam positions. Seams: mean 2.74 / median 2.09 mm. Random positions on the
same terrain: mean 3.68 / median 2.69 mm, max 23.4 where real faults are. Seams
are *quieter* than ordinary sidewalk. A few millimetres sounds alarming until
you learn the surface does more than that everywhere.

**Import scaling — resolved 2026-09-14.** A 234 m walkway at 1 cm (4.2 M
vertices, 8.4 M triangles) **froze Isaac outright**. Chunking into 20 m
collider prims fixed the PhysX side but not the freeze: the binding constraint
is the **RTX viewport**, which draws every triangle regardless of how many
prims they sit in, while the import awaits the app loop that rendering blocks.

Mitigations now in `import_terrain_isaac.py`:
- `MAX_TRIANGLES = 2_000_000` refuses an oversized import and names the fix.
- `RENDER_TERRAIN = False` hides terrain from the renderer, collision intact —
  the right setting for physics-only collection runs, where Spot walks a known
  straight strip and needs no camera.
- `CHUNK_LENGTH_M = 20.0` splits the walkway; adjacent chunks **share** their
  boundary column, or feet catch in the hairline gap between coincident edges.
- `PLACE_SPOT` stands the robot at the walkway start, so every run begins from
  the same pose and runs stay comparable.

Working configuration: **70.6 m at 2 cm cells** (321 k vertices, 635 k
triangles, 4 chunks, 6 segments, 697 windows). 2 cm is finer than Spot's ~4 cm
foot, so nothing measurable is lost. Verified: 400 rays, 0 missed, **max error
0.000 mm**.

Two bugs found during that verification, both worth remembering:
- Placing Spot on the surface *before* verifying it meant one sample ray hit
  the robot, at exactly the ray-origin height. One 1000 mm outlier among 399
  exact rays averaged to **+2.500 mm** — small enough to look like a plausible
  systematic calibration error. The max, not the mean, exposed it. Verification
  now walks rays past the robot, as the evenness sampler already did.
- `ClearXformOpOrder()` + `AddTranslateOp()` destroyed Spot's existing rotate
  ops and produced a non-orthonormal transform. Use
  `UsdGeom.XformCommonAPI.SetTranslate()`.

## Phase 3 — Collection with domain randomization

Sim-only training makes randomization the entire transfer story.

- **Terrain** — the Phase 1 parameter space, sampled to balance severity.
- **Sensor** — noise, bias, drift, rate, latency, mounting offset.
- **Robot** — mass, inertia, payload, actuator strength, joint friction.
- **Contact** — friction, restitution.
- **Motion** — commanded speed, and heading relative to the defect. Approach
  angle matters: crossing a faulted joint square is a different signal from
  crossing it obliquely.
- **Gait / controller parameters** — as aggressively as the policy tolerates.
  This is the mitigation for the controller gap, and it is the highest-risk
  item in the plan.

Lock commanded speed per run rather than letting perception set it. The current
speed schedule in the walker keys off sidewalk width and centerline alignment,
which correlates speed with scene geometry and confounds the roughness signal.

---

## Phase 4 — Model

- **Regress, don't classify.** Predict detrended peak-to-peak, RMS, and maximum
  vertical discontinuity as continuous targets; threshold at evaluation time.
  Better-conditioned loss, re-binnable, and it sidesteps the class imbalance
  that makes "always predict compliant" a strong classifier.
- **Baseline before deep learning.** Windowed hand features — RMS, peak-to-peak,
  spectral centroid, contact-timing variance, touchdown height variance — into
  gradient boosting. Get a number first. A 1D temporal CNN that cannot beat it
  is telling you the problem is data, not architecture.
- **Ablate by modality**: foot contact geometry only / + joint state / + IMU /
  IMU only. This ablation is itself a result, and it identifies which signals
  are worth fighting for on the real robot.
- **Split by parameter region, never by random window.** 1 m windows at 10 cm
  stride overlap 90%; a random split puts near-identical windows on both sides
  and reports a fake number. Hold out severity ranges to measure extrapolation.

---

## Phase 5 — Real validation

Not a training set. A check that transfer happened, sized accordingly.

**Calibrated-shim protocol.** Lay boards of known thickness on flat real
pavement and walk Spot over them at a fixed gait and speed. Hardware-store
plywood brackets the ADA thresholds closely. Exact real-world ground truth, no
surveying, roughly $30 and an afternoon — and it tests precisely the decision
boundary the model is trained on.

Report three numbers: sim-only, real-only (if any real training data ever
exists), and sim-trained-evaluated-on-real. The gap between the first and third
**is** the sim-to-real result.

**Optional extension.** If a surveyed real corridor becomes available (rod and
level, rolling dipstick, or terrestrial laser scan), 50–200 m spanning the
severity range is enough for a continuous-profile evaluation.

---

## Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Controller gap** — sim policy ≠ BD controller, and terrain response is mediated by the controller | High | Randomize gait/controller params; weight features toward controller-invariant foot geometry; pilot the shim test early rather than after a full collection campaign |
| Sim terrain distribution doesn't cover real sidewalks | High | Parametric generation with ranges drawn from pavement-engineering literature, not from what the 17 assets happen to contain |
| Spot's controller rejects the disturbance so well that little signal reaches any sensor | Medium | Foot touchdown height is upstream of rejection; the ablation will reveal this early |
| Real state rate/latency worse than assumed | Medium | Phase 2 measures it before any model is trained |
| Overfitting to authored terrain regularities (exact slab spacing, symmetric domes) | Medium | Jitter every structural parameter; hold out parameter regions |

---

## Success criteria

1. Detect a 13 mm vertical discontinuity on real hardware, trained only in sim,
   at a usable false-positive rate.
2. Separate the 6.4 mm and 13 mm ADA classes better than chance on real data.
3. Report the modality ablation, so the result says *which* sensing matters.

Criterion 1 is the paper. Criteria 2 and 3 are what make it useful to someone
assessing sidewalk compliance.

---

## Status

- [x] Design settled, decisions recorded
- [x] Phase 1 — generator, labeller, Isaac importer
- [x] Phase 1 exit — raycast matches authored heights to 0.000 mm (2026-09-14)
- [x] Phase 2 sim half — IMU, foot contact/position, joint state all verified
- [ ] Phase 2 real half — record the SDK data contract (blocked on Spot access)
- [ ] **Phase 3 — batch collection runner** ← current
- [ ] Phase 2 — real data contract + mirrored sim logging
- [ ] Phase 3 — randomized collection
- [ ] Phase 4 — baseline, then model, then ablation
- [ ] Phase 5 — shim validation on real Spot
