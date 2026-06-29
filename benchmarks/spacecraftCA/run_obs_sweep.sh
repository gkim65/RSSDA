#!/bin/bash
# run_obs_sweep.sh — SDec graded-obs σ sweep over the 3 spherical case-study conjunctions.
# Each cell: solve (drag backend) -> brahe-MC rollout -> save the policy -> log to wandb + CSV.
#
# USAGE (from anywhere; cd's to its own dir):
#   bash run_obs_sweep.sh                     # foreground
#   nohup bash run_obs_sweep.sh &             # survive SSH disconnect; tail nohup.out
#
# PRE-REQS on the box: deps installed (requirements.txt), `wandb login` done (or set
#   WANDB_OFFLINE=1 below), brahe EOP data available (drag needs it).
#
# WHAT YOU GET PER CELL (tag = <case>_drag_sig<σ>):
#   notes/results/variant_*_<tag>.csv        return / per-stage burns / action schedule / reward parts
#   notes/results/rollout_<tag>_sdec.csv     per-rollout true brahe miss + 4-7km band stats
#   notes/policies/<tag>__sdec__bin0.pkl     the solved policy (re-fly later, ZERO re-solve; ~KB, light)
#   logs_<tag>.txt                           full stdout (per-iter [A*] progress is OFF here; see VERBOSE)
set -u
cd "$(dirname "$0")"                      # -> benchmarks/spacecraftCA
V=../../.venv/bin/python                  # adjust if the cluster python differs

# ---- knobs you may want to tweak --------------------------------------------
SIGMAS=(0.3 0.5 0.7 1.0 1.5 2.0)         # low -> high: tractable cells log FIRST
CASES=(head_on oblique cross_track)
INIT_MISS=0.5                            # belief center (km). See note below re: straddling.
INIT_SPREAD=1.4                          # belief spread (km).
ITER_LIMIT=50000                         # TI2 budget -> a hung cell BAILS with a number (partial result)
MEM_LIMIT_GB=100                         # RS-SDA* memory ceiling (GB). Raise to your box; <=0 = no limit.
ROLLOUTS=200                             # brahe-MC rollouts per cell
WANDB_PROJECT=spacecraftCA-obs-sweep
# WANDB_OFFLINE=1                        # uncomment if you can't `wandb login`; `wandb sync` later
# -----------------------------------------------------------------------------

SC1='[6928136.3,0.001,55.0,20.0,0.0,0.0]'
declare -A SC2
SC2[head_on]='[6927709.152731867,0.013204300866479831,55.00000000000001,19.99999999999998,86.69478206051107,274.8562126354506]'
SC2[oblique]='[6815932.893497958,0.01852077796054433,55.0,19.99999999999998,150.0427595370889,210.99970849556453]'
SC2[cross_track]='[6920768.808233308,0.0007859467086336181,55.00000000000001,19.99999999999998,179.99999999999736,180.00000000000264]'

wandb_flags="wandb.enabled=true wandb.project=${WANDB_PROJECT}"
if [ "${WANDB_OFFLINE:-0}" = "1" ]; then wandb_flags="${wandb_flags} wandb.mode=offline"; fi

echo "######## obs σ sweep START $(date) ########"
echo "sigmas: ${SIGMAS[*]} | cases: ${CASES[*]} | init_miss=${INIT_MISS} spread=${INIT_SPREAD} | iter_limit=${ITER_LIMIT} | rollouts=${ROLLOUTS}"

for SIG in "${SIGMAS[@]}"; do
  for CASE in "${CASES[@]}"; do
    TAG=${CASE}_drag_sig${SIG}
    LOG=logs_${TAG}.txt
    echo ""
    echo "===== $CASE  sigma=$SIG  -> tag=$TAG  start $(date) ====="
    $V -u main.py \
      solve.variants='[sdec]' grid.propagator=drag obs.sigma=$SIG \
      conjunction.sc1_oe="$SC1" conjunction.sc2_oe="${SC2[$CASE]}" \
      belief.init_miss=${INIT_MISS} belief.init_spread=${INIT_SPREAD} \
      solve.sdec_iter_limit=${ITER_LIMIT} solve.sdec_memory_limit_gb=${MEM_LIMIT_GB} \
      run.rollout=true run.rollouts=${ROLLOUTS} run.save_policy=true \
      ${wandb_flags} \
      wandb.name=${TAG} run.tag=${TAG} \
      2>&1 | tee "$LOG"
    echo "===== $CASE  sigma=$SIG  end $(date)  (exit ${PIPESTATUS[0]}) ====="
  done
done

echo ""
echo "######## obs σ sweep END $(date) ########"
echo "results -> notes/results/   policies -> notes/policies/   logs -> logs_*.txt"
