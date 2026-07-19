# Figures — how to recreate

Index of the paper/slide figures, the script that generates each, and the exact command.
Output goes to `benchmarks/spacecraftCA/notes/figures/`.

> **⭐ Figures used in the FINAL PAPER** (sections marked ⭐ below):
> - `reward.pdf` — two-ramp terminal reward → [`plot_reward.py`](../plot_reward.py)
> - `sweep52_coverage.pdf` — 52-conjunction coverage → [`plot_sweep52_coverage.py`](../plot_sweep52_coverage.py)
> - `summary_violin_all9_boxonly.pdf` — planners vs operator floor → [`plot_summary_violin.py`](../baselines_spacecraftCA/plot_summary_violin.py)

> **Note on `plot_v2_concept.py`:** the paper's reward figure was factored out into the
> standalone [`plot_reward.py`](../plot_reward.py) (committed). `plot_v2_concept.py` still holds
> the schematic *explainer* panels (`concept_1/2/3`), which are slide-only and contain glyphs
> (✓ ✗ °) that break under `text.usetex=True` — clean those before committing that script.
> When committing figures, scope it to ONLY the ones we actually want (don't blanket-add every
> PNG/PDF/SVG in `notes/figures/` — many are scratch/experiment outputs).

## Conventions (all figures)

- **Run from the repo root** using the project venv: `.venv/bin/python`.
- **Typography:** scripts auto-detect a system LaTeX install and render text in **Computer
  Modern** via `text.usetex=True` (falls back to matplotlib's mathtext-cm if `latex` is not
  on PATH). No hard-coded titles on the paper figures — captions live in the paper.
- **Formats:** the shared `_save()` helper writes a PNG plus vector copies (PDF for the
  paper, SVG for slides/web). Vector files stay sharp at any zoom.
- **Transparent background:** paper/slide figures are saved with `transparent=True` (figure,
  axes, and legend frame all alpha-0), so they drop onto any slide/page color.
- **`--` in usetex** renders as an en-dash (–); intentional in labels like "1km zone -- maximum penalty".
- **Label capitalization:** every user-facing label — axis labels, titles, legend entries, and
  floating annotations — starts with a **capital first letter** (sentence case), e.g.
  "Miss distance at TCA (km)", "Rollouts", "Ideal zone at TCA", "Initial 1 km". Only the first
  word is capitalized (not title case). Variant names that are raw data values (`sdec`, `dec`,
  `centralized`) are left lowercase in legends. Underscore-laden field names must be mapped to a
  clean label (see `_COL_LABELS` / `_col_label` in `plot_rollout_dist.py`) rather than shown raw.

---

## `appendix_perp_frozen` + `appendix_reduced_miss` — state-space approximation validation

**What they show:** the two brahe-backed appendix figures that justify the reduced state
space (the state-space section's two load-bearing approximations). Both use **full 6-D RTN
propagation to TCA as ground truth** (via `brahe`), reusing the model's own machinery
(`spacecraft_transition_v2`, `spacecraft_matrices`, and `brahe_miss_sequence` imported from
`plot_v2_accuracy`), so the figures **cannot silently drift** from the model.

**Script:** [`plot_appendix_validation.py`](../plot_appendix_validation.py)

**Command** (run from repo root; each figure writes `.pdf` + `.svg` + `.png`):
```bash
# light theme (paper); default burn magnitude = model DV (0.5 m/s)
.venv/bin/python -u benchmarks/spacecraftCA/plot_appendix_validation.py
# dark theme (black slides)
.venv/bin/python -u benchmarks/spacecraftCA/plot_appendix_validation.py --dark
# override the burn magnitude used in Appendix A (m/s)
.venv/bin/python -u benchmarks/spacecraftCA/plot_appendix_validation.py --dv 0.5
```

### `appendix_perp_frozen` — Approximation A: along-track burns barely move `perp`
**Single panel** (no suptitle). Fires one 0.5 m/s along-track burn (both signs) at every
decision stage, propagates each to TCA with brahe, and plots the along-track change
`|Δ δpT|` (blue) and the perpendicular change `|Δ perp|` (purple), where
`perp = √(pR² + pN²)`, against **time-to-TCA (days)** on a log y-axis. Along-track rides at
10–100 km; perp hugs ~1 km — an order-of-magnitude+ gap that reads directly. The dashed 1 km
collision line is a scale reference only. Boxed legend sits in the **lower-right**, clear of
all data (dark-theme-aware facecolor via `_IS_DARK`).
- Console prints max `|Δ δpT|`, max `|Δ perp|`, and the median leak ratio
  `|Δ perp| / |Δ δpT|` ≈ 1.8 % (perp responds ~57× more weakly).
- **HONEST CAVEAT (do not undo):** perp is **not** literally sub-km for the earliest burns
  (max `|Δ perp|` ≈ 2.55 km over the 24 h lever). The defensible claim is that perp is only
  weakly, higher-order affected — matching the paper's "weakly affected through higher-order
  coupling … do not contribute significantly to the leading-order variation." **Do NOT add a
  "perp stays below the collision threshold" caption/title** — it is false for the earliest
  burns and an earlier draft wrongly asserted it. (An earlier version also had a right-hand
  leak-ratio panel; removed at Grace's request — the single panel carries the story.)

### `appendix_reduced_miss` — Approximation B: reduced miss ≈ true 3-D brahe miss
**Two panels, no suptitle.** Fires a single burn (`+δv` AND `-δv`) at each stage over a
spread of conjunction geometries `(perp, δpT0)`, comparing the reduced-model miss
`m = √(δpT² + perp²)` against the true 3-D brahe miss `|r_RTN(TCA)|`.
- **Left panel:** reduced miss vs brahe miss on a `y = x` line (symlog both axes) — points hug
  the diagonal across ~4 orders of magnitude (sub-km to ~130 km). Boxed legend top-left.
- **Right panel:** absolute miss error for a single burn at **each stage**, `+δv` (blue) and
  `-δv` (purple) bars side by side, averaged over the geometries. All stages sit in ~0.25–2 km;
  the point is that the error is low for a burn fired at ANY time in either direction (no
  cherry-picked sequences).
- **Terminal stage excluded** (`stages = range(N_STAGES - 1)`): a "burn" one step before TCA
  has ~zero lever, so it is a degenerate decision and its mean-rate reduced approximation is
  worst there (~7 km); including it dominated the y-axis. Do not re-add it.
- Console prints mean/max error and the worst `+δv` / `-δv` stage.

**Key implementation notes:**
- Nothing is cached: Appendix A runs 50 brahe propagations, Appendix B runs 192 (4 geometries
  × 24 stages × 2 signs); ~1–3 min total, brahe propagation is the bottleneck. If model params
  change (stage grid, DV, geometry), just rerun.
- Appendix A's conjunction anchor is `perp0 = 3.0 km, δpT0 = 1.0 km`; the claim is
  geometry-independent but the quoted numbers are for this anchor (edit `perp0_km, dt0_km`
  in `plot_perp_frozen` to re-anchor).
- Follows the shared conventions: sentence-case labels, Computer Modern via usetex
  (mathtext-cm fallback), `--dark` theme flag, PNG+PDF+SVG saved `transparent=True`.

---

## `reward.pdf` — two-ramp terminal reward (v2)  ⭐ USED IN THE FINAL PAPER

**What it shows:** the v2 terminal reward as two opposing ramps vs miss distance at TCA —
a **risk ramp** (near-field collision penalty, floor −10000, cleared by 5 km) and a
**displacement ramp** (convex return-to-slot cost past the 5 km station-keeping tube). The
two hand off at the **5 km optimum** ("clear & stop"). Plotted on a **signed-log** y-axis so
the −10000 risk floor and the shallow displacement bowl are both legible on one axis; a side
**zoom panel** (x 4–7 km, reward 0…−2) details the handoff. A dotted rectangle on the main
panel marks the zoom window.

**Script:** [`plot_reward.py`](../plot_reward.py) — a self-contained script (factored out of
`plot_v2_concept.py`, which also holds schematic explainer panels **not** used in the paper).

**Command:**
```bash
.venv/bin/python benchmarks/spacecraftCA/plot_reward.py
```
(one call writes the paper PDF + PNG/SVG + all black-slide variants).

**Outputs (one call writes all):**
- `reward.pdf` — **the paper figure**, annotated, normal line weight, transparent.
- `concept_4_two_ramp_reward.png` + `.pdf` — same figure (PNG raster + PDF vector).
- `concept_4_two_ramp_reward.svg` — **thicker lines (s=1.8)** for slides/web.
- `reward_black.pdf` + `.svg` — **black-slide version** (`dark=True`): white foreground
  (text/ticks/spines/grey-total/zoom-rectangles), transparent background. Same colored
  ramps. PDF at normal weight, SVG at s=1.8. Drop onto a black slide.
- `reward_black_bg.pdf` — same white-on-black figure but with the **black background baked
  in** (opaque), for placing on a non-black page. Use `reward_black.*` instead when the
  slide already provides the black.

**Annotations (replicate Grace's hand-marked version):**
- Green **optimum star** at (5 km, 0) on both the main and zoom panels + green
  "Optimum / Reward at 5km" callout with arrow.
- Magenta dashed vertical line at **1 km** labeled "1km zone -- maximum penalty" (rotated).
- Green dotted vertical line at **5 km** labeled "5km cleared / screening zone".

**Key implementation notes:**
- Reward curves are pulled **live** from `spacecraft_transition_v2.py`
  (`risk_ramp_reward`, `displacement_cost`, and the RISK_/DISP_ constants) via `_reward_fns()`,
  so the figure **cannot drift** from the model. If the reward changes, just re-run.
- `_signed_log(r) = sign(r)·log10(1+|r|)`; y-tick labels show the REAL values
  (0, −1, −10, −100, −1000, −10000).
- Total reward is drawn **behind** (thick grey) the red risk + purple displacement ramps.
- Line thickness is controlled by the `s` scale arg in `_build(s=...)` (1.0 paper, 1.8 SVG).
- The green 5 km line uses `C_BURN`; the magenta 1 km line uses `C_PEN`.

**History:** the pre-Mahdi version hard-coded a −100 risk floor + a *linear* displacement
ramp; the real v2 reward is a −10000 floor + a *quadratic* displacement bowl (see
`spacecraft_transition_v2.py` DISP_QUADRATIC_K). That mismatch is why the figure was
redesigned onto a signed-log axis. See `MODEL_DEFINITION.md` for the reward spec.

---

## Other panels in `plot_v2_concept.py` (schematic, no data)

`plot_v2_concept.py` also has three schematic panels (run each function, or the whole set
via `python plot_v2_concept.py`). NOTE: some of these still contain glyphs (✓ ✗ ° %) that
break under `text.usetex=True`; sanitize before a full `__main__` run with LaTeX on.

- `concept_1_state_decomposition` — `plot_state_decomposition()` — (δT, perp) geometry: miss = hypotenuse.
- `concept_2_v1_bug` — `plot_v1_bug()` — burn+counterburn in the (δT, vdev) phase plane.
- `concept_3_conjunction_types` — `plot_conjunction_types()` — head-on / oblique / cross-track via perp.
- `concept_0_overview` — `plot_combined()` — 2×2 sheet of all four panels (the three schematic
  ones plus the reward panel; the reward panel itself is the paper figure, see `plot_reward.py`).

---

## `rollout_miss_shift_overlay[_pooled][_black]_<tag>[_<filter>].{png,pdf,svg}` — initial→final miss

**What it shows:** the per-rollout distribution of **final miss distance at TCA** for each
variant (centralized / SDec / Dec), against the **initial** miss the conjunctions started at.
The paper claim reads straight off it: **SDec (all 6 GS contacts) recovers the centralized
policy** — the blue `\\\` centralized hatch rides exactly on top of the solid green SDec
fill — while **Dec (no sync) overshoots and disperses**, its red mass shifted right of the
green/blue and worst for close (1–2 km) conjunctions.

Two layouts (same function, `--pool` toggles):
- **faceted** — one panel per starting-miss family (1/2/5/10 km), so each start level's
  before→after is separate.
- **pooled** (`--pool`) — all starting misses collapsed into one panel.

**Script:** [`plot_rollout_dist.py`](../plot_rollout_dist.py) → `plot_miss_shift_overlay()`

**Commands** (run from repo root; reads the sweep's `--save-rollouts` `.npz` dumps in
`notes/results/rollouts_<tag>/`):
```bash
J=benchmarks/spacecraftCA/notes/conj_sweep_spherical_50.json
# faceted, init_miss=0.5 belief only:
.venv/bin/python benchmarks/spacecraftCA/plot_rollout_dist.py \
  --tag sweep50_drag --miss-shift-overlay --conj-json $J --filter init_miss=0.5
# pooled:
.venv/bin/python benchmarks/spacecraftCA/plot_rollout_dist.py \
  --tag sweep50_drag --miss-shift-overlay --pool --conj-json $J --filter init_miss=0.5
# black-slide theme (white foreground, transparent bg -- drops onto any slide):
.venv/bin/python benchmarks/spacecraftCA/plot_rollout_dist.py \
  --tag sweep50_drag --miss-shift-overlay --conj-json $J --filter init_miss=0.5 --dark
# same, but with the black background BAKED IN (opaque) for a non-black page:
.venv/bin/python benchmarks/spacecraftCA/plot_rollout_dist.py \
  --tag sweep50_drag --miss-shift-overlay --conj-json $J --filter init_miss=0.5 --dark --bg
```

**No fixed-belief sweep required** — for a `--init-miss true` ("true belief": each
conjunction's belief centered on its own true miss, e.g. `truebelief_spherical50`) sweep, drop
`--filter init_miss=...` entirely; there is no single fixed belief-center to slice on, so just
pass the whole tag through and let `_fam` facet on the conjunction's actual starting `miss_km`:
```bash
.venv/bin/python benchmarks/spacecraftCA/plot_rollout_dist.py \
  --tag truebelief_spherical50 --miss-shift-overlay --conj-json $J
```

**Styling (per-variant, in `_VARIANT_STYLE` / `_VARIANT_COLORS` / `_VARIANT_LABELS`):**
- **dec** — solid fill, **maroon** (`#7f0000`) outline (its overshoot reads as a block).
- **sdec** — solid green fill, **dark-green** (`#0b4d0b`) outline.
- **centralized** — **blue `\\\` hatch only** (clear background, 0.8-alpha) + a blue
  short-dashed outline, drawn LAST on top so it visibly traces over SDec where the two
  coincide.
- Legend uses **full names** (`_VARIANT_LABELS`: Decentralized / Semi-Decentralized /
  Centralized), no `(n=...)` counts.
- Grey dashed **initial-miss** line per panel at its family value, labeled **inline** ("N km",
  rotated) rather than in the legend; **no** collision line. The labeled green **"ideal zone at
  TCA"** 4–7 km band is the reference.
- **Panel roles:** legend on the **1st** panel, "ideal zone at TCA" label on the **2nd**, so the
  two don't overlap. Panel title is "Initial miss: N km".
- **`--dark`** (per FIGURES.md's shared dark-slide convention): flips text/ticks/spines/legend
  to **white**, lightens the initial-miss reference line/label greys, and brightens the "Ideal
  zone at TCA" annotation to a pale green — **variant fill colors are unchanged** (only
  foreground/background inverts). Saves `transparent=True` by default (drops onto any slide);
  add **`--bg`** to instead bake in an **opaque black** background (matching the
  `reward_black.pdf` vs `reward_black_bg.pdf` pattern) for placing on a non-black page. Output
  filename gets a `_black` suffix (before the tag) when `--dark` is set.

**Layout / sizing:**
- Faceted figure is `~4.0 in/panel` (≈16 in for 4 across) at a large base font (`fs≈20`) so it
  reads like ~11 pt once **scaled down ~2× onto an 8.5×11 page**. Fonts scale off `fs`.
- Fig height 5 in with the count y-ceiling pinned to **1000** (`set_ylim(top=1000)`) so the
  legend clears the tallest bars (bars peak ~650 for the 2600-rollout sweep). NOTE: that 1000 is
  tuned to this sweep's rollout count — bump it for a larger sweep. Density mode keeps auto range.

**Key implementation notes:**
- The **initial spread** comes from [`conj_initial_miss.py`](../conj_initial_miss.py), which
  recomputes each conjunction's true no-maneuver closest-approach miss from its orbit sets
  (`--conj-json`). Snapped to the 4 design families via `_nearest_family`. `_shared_edges` floors
  the left x-edge at `(smallest initial − 0.5)` so the 1 km initial line isn't clipped.
- `--filter` (e.g. `init_miss=0.5`) subsets cells before loading AND is folded into the saved
  filename, so different-belief runs don't clobber each other.
- Typography + saves match the paper convention here too: `_setup_typography()` (Computer
  Modern via usetex, mathtext-cm fallback) and `_save()` writing PNG + PDF + SVG. Axis labels
  are prettified via `_COL_LABELS` and `_tex_safe()` (escapes underscores under usetex).

---

## `peel_heatmap_<tag>[_<conj>].{png,pdf,svg}` — greedy contact peel-down

**What it shows:** the greedy sync-contact peel-down for one conjunction, as a two-panel
figure. **Left** = a heatmap whose columns span **all** stages `0 .. N_STAGES-1` (T−24h→TCA)
and whose rows are the SDec subsets the search tried, **top-to-bottom in greedy order**
(window → each `greedy_drop*` → `minimal`); a shaded cell marks a **kept** sync contact, so
you watch the contacts get peeled away against the sparse full grid. Red ▼ markers over the
columns flag the GS-contact stages available for that conjunction. **Right** = a bar per row
for **expected return**, with the numeric return + collision % printed at each bar end, a
dashed **Centralized rail** and its **±tol band**, and a dotted **Dec rail** for reference.
The story: return holds on the Centralized rail as contacts drop — until (under a loose tol)
it doesn't, which is exactly how the figure exposes over-peeling.

**Script:** [`plot_peel_heatmap.py`](../plot_peel_heatmap.py)

**Data source — two ways** (same figure either way):
- **wandb (preferred)** — pulls one run per solved subset from the peel run's project
  (config carries `contacts`/`subset_name`/`variant`/`label`, metrics carry
  `expected_return`/`collision_prob_matrix`/`n_stages`); runs are ordered by creation time to
  recover the greedy sequence. This also fixes the stage count: `n_stages` comes from the run,
  so all ~24+ columns show even where the model can't be imported locally.
- **local CSV** — `--source csv --csv notes/results/peel_<tag>.csv` (peel_contacts.py's output).

**Commands** (run from repo root):
```bash
# From wandb — the tol=5 exploration run (over-peels; entity/project defaults shown explicitly):
.venv/bin/python benchmarks/spacecraftCA/plot_peel_heatmap.py \
  --source wandb --wandb-entity kmeans_gsopt --wandb-project spacecraftCAsyncs \
  --tag syncs_tol5
# The exact-match tol=0.001 run — just swap the project (and give it its own tag):
.venv/bin/python benchmarks/spacecraftCA/plot_peel_heatmap.py \
  --source wandb --wandb-entity kmeans_gsopt --wandb-project spacecraftCAsyncsTol \
  --tag closemiss_tol001
# From a local peel CSV instead:
.venv/bin/python benchmarks/spacecraftCA/plot_peel_heatmap.py \
  --source csv --csv notes/results/peel_peel_headline.csv --tag peel_headline
```
One figure per conjunction (`label`) is written unless `--conj <label>` filters to one.

**Useful flags:** `--wandb-tag <group>` filters to a single peel run's `group` when a project
holds several; `--theme light|dark` picks foreground ink for white vs black backgrounds;
`--tol` sets the width of the ±tol band drawn at the rail (default 0.001, match the peel run);
`--n-stages N` forces the column count when the model can't be imported and the data lacks an
`n_stages` field (normally auto: model → data → this override).

**Key implementation notes:**
- `N_STAGES` resolves model (`spacecraft_stage_grid.N_STAGES`) → data-derived → `--n-stages`.
  On a box without `brahe`/the model, the wandb/CSV `n_stages` keeps the full grid.
- Rows preserve source order (CSV row order, or wandb `created_at`) so the peel reads as a
  sequence; `centralized`/`dec` rows become the rail reference lines, not heatmap rows.
- Follows the shared conventions: sentence-case labels with raw field names mapped to clean
  ones, Computer Modern via usetex (mathtext-cm fallback), PNG+PDF+SVG saved `transparent=True`.

**See also:** [`peel_pseudocode_draft.md`](peel_pseudocode_draft.md) for the algorithm this
figure visualizes, and the peel run command in the top-level `CLAUDE.md`.

---

## `sweep52_coverage.{png,pdf,svg}` — the 52-conjunction evaluation suite  ⭐ USED IN THE FINAL PAPER

**What it shows:** the full evaluation suite as the ECI orbit of the secondary spacecraft
(SC2) in each of the 52 conjunctions, in three projections (**XY / XZ / YZ**). The Earth is
the gray disk, the SC1 reference orbit is black, and each colored ellipse is one SC2 orbit
we conjunct with, **colored by plane-crossing angle Δi = |i₂ − i₁|** (dark blue: near-coplanar
co-orbital; red: steep / near-retrograde crossings). Arcs behind the Earth are faint-dashed
(same occlusion cue as Figure C/D in `plot_coverage_scope.py`). It is the standalone version
of the D6 "representative sweep" panel of `coverage_5dof_spread.png`, over the reported
`conj_sweep_spherical_50.json` (52 scenarios). **No on-figure caption** — the paper caption
lives in the paper.

**Script:** [`plot_sweep52_coverage.py`](../plot_sweep52_coverage.py)

**Command** (run from `benchmarks/spacecraftCA/`; needs `PYTHONPATH=.` because it imports the
occluded-ellipse helpers from `plot_coverage_scope` and `SC1_OE_AT_TCA` from `spacecraft_matrices`):
```bash
PYTHONPATH=. ../../.venv/bin/python -u plot_sweep52_coverage.py
# black-background slide variant (light foreground):
PYTHONPATH=. ../../.venv/bin/python -u plot_sweep52_coverage.py --dark
# a different suite / output stem:
PYTHONPATH=. ../../.venv/bin/python -u plot_sweep52_coverage.py \
  --json notes/conj_sweep_spherical_50.json --out sweep52_coverage
```

**Key implementation notes:**
- The suite spans four screening-miss levels {1,2,5,10} km, Δi up to ~115°, encounter angles
  3°–90°, and altitudes {550,800,1200} km; all SC2 orbits are near-circular (e ≤ 0.005).
- **Color axis is the naive Δi = |i₂ − i₁|** (scale 0–120°), matching D6 and the paper's
  "Δi up to ~115°" wording. The *true* plane-to-plane angle (RAAN-aware,
  `plot_coverage_scope.plane_to_plane_angle`) reaches ~135° — swap to it if the paper prefers.
- Reuses `koe_ellipse_eci` / `earth_circle` / `_draw_occluded` / `_PROJ` from
  `plot_coverage_scope.py`, so the orbit-ellipse rendering can't drift from Figure C/D.
- Follows the shared conventions: sentence-case labels, Computer Modern via usetex
  (mathtext-cm fallback), `--dark` theme flag, PNG+PDF+SVG saved `transparent=True`.

---

## `summary_violin_all9_boxonly.{pdf,svg,png}` — planners vs operator-heuristic floor  ⭐ USED IN THE FINAL PAPER

**What it shows:** the headline comparison of miss-distance-at-TCA distributions for all nine
strategies on one horizontal axis, grouped into two blocks: the three **optimized planners**
(Centralized / Semi-decentralized / Decentralized) on top and the six **representative operator
heuristics** below. Each row is an exact box (q1/median/q3), p05–p95 whiskers, a white median
line and a ◇ mean; three shaded zones mark **Unsafe (<4 km)**, the **Safe / target band
(4–7 km)**, and **Over-mitigated (>10 km)**, with each row's "% in target band" printed in a
clear zone just to the right of the widest box (so the callouts never cover a box plot). The
figure is a wide 16:9 landscape with large fonts sized to read at the placed size. The operator
heuristics are ordered self-play first (Threshold×threshold,
Selfish×selfish, Fixed-lead×fixed-lead — maroon→orange family) then cross-play
(Threshold×selfish, Threshold×fixed-lead, Selfish×fixed-lead — purple family). Story: the
planners cluster in the green target band (~60/61/28 % in-band) while every operator heuristic
sprawls into the over-mitigated zone (~17–25 % in-band).

**Script:** `baselines_spacecraftCA/plot_summary_violin.py`. It is **stats-driven** — it reads a
JSON summary (per category: n, mean, min, q1, median, q3, max, p05, p95, coll_pct, band_in_pct),
NOT the raw rollouts. The box is exact; the optional violin is *reconstructed* from the reported
percentiles (so the box is the truth — prefer `--no-violin` for the paper). A copy of the current
numbers is embedded as the default `STATS` dict in the script, so it runs with no `--json`.

**Regenerate the stats** (from the cluster/local `.npz` dumps) with `summarize_npz.py`, which
prints a paste-ready JSON for both the operator-matrix and POMDP-rollout `.npz` layouts:

```bash
cd benchmarks/spacecraftCA
# operator baselines (opmatrix run):
../../.venv/bin/python -u baselines_spacecraftCA/summarize_npz.py \
  --dir notes/results/opmatrix_opmatrix50/npz --json-out stats_ops.json
# POMDP variants (sweep run with --save-rollouts):
../../.venv/bin/python -u baselines_spacecraftCA/summarize_npz.py \
  --dir notes/results/rollouts_<sweep_tag> --json-out stats_pomdp.json
# (merge the two JSON objects into one, or paste both into the STATS dict.)
```

**Exact command for the paper figure** (box-only, words-above band key, from embedded STATS):

```bash
cd benchmarks/spacecraftCA
../../.venv/bin/python -u baselines_spacecraftCA/plot_summary_violin.py \
  --tag all9_boxonly --band-legend arrows --no-violin
# from a JSON instead of the embedded default:
../../.venv/bin/python -u baselines_spacecraftCA/plot_summary_violin.py \
  --json stats.json --tag all9_boxonly --band-legend arrows --no-violin
```

Output: `notes/figures/summary_violin_all9_boxonly.{png,pdf,svg}`. Useful flags: drop
`--no-violin` for the reconstructed-violin version; `--band-legend {arrows,corner,strip}`;
`--dark` / `--transparent` for slides; `--ymax` to cap the x-axis.

**All variants currently in `notes/figures/`** (the output filename is `summary_violin_<tag>`,
so pick the tag to match; `--transparent` = drops onto any bg, `--dark` = white foreground for
black slides):

```bash
cd benchmarks/spacecraftCA
PY=../../.venv/bin/python
# paper box-only (light) and its black-slide twin:
$PY -u baselines_spacecraftCA/plot_summary_violin.py --tag all9_boxonly      --no-violin --transparent
$PY -u baselines_spacecraftCA/plot_summary_violin.py --tag all9_boxonly_dark --no-violin --dark --transparent
# reconstructed-violin versions, one per band-key style:
$PY -u baselines_spacecraftCA/plot_summary_violin.py --tag all9_arrows  --band-legend arrows  --transparent
$PY -u baselines_spacecraftCA/plot_summary_violin.py --tag all9_strip   --band-legend strip   --transparent
$PY -u baselines_spacecraftCA/plot_summary_violin.py --tag all9_corner  --band-legend corner  --transparent
$PY -u baselines_spacecraftCA/plot_summary_violin.py --tag all9_partial --band-legend arrows  --transparent
```

---

## TODO / not yet documented here

Add entries as figures stabilize. Candidate scripts already in the tree:
`plot_v2_accuracy.py`, `plot_policy.py`, `plot_conjunction_geometry.py`,
`plot_rtn_to_state.py`, `plot_trajectory.py`, `plot_coverage_scope.py`,
`plot_sweep_coverage.py`, `replot_from_csv.py`.
