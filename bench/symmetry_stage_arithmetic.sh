#!/bin/bash
# The stage arithmetic, broadcast against by owner, on one Symmetry AMD node
# (64-core EPYC 7543, 8 NUMA domains of one CCD each), for "Stage
# arithmetic" in CODE.md (step 5 of PLAN.md). After TreeAMR's
# bench/symmetry_affinity.sh (branch claude/festive-bun-656842).
#
#     sbatch bench/symmetry_stage_arithmetic.sh
#
# from a checkout of this package; IMEXRK_REPO overrides the directory.
# Each run is bench/stage_arithmetic.jl's thread sweep, one fresh Julia
# process per thread count, on a 10^8-byte Float64 state. Three memory
# placements:
#
#     pinned, first touch   JULIA_EXCLUSIVE=1, no numactl: `init` writes
#                           the scratch by owner, so each thread's pages are
#                           on its own domain (the broadcast's are wherever
#                           the one thread that first touched them runs)
#     pinned, interleaved   JULIA_EXCLUSIVE=1, numactl --interleave=all
#     unpinned, first touch no JULIA_EXCLUSIVE: the OS may move threads
#
# Pinned runs use `--threads=n,0`: from Julia 1.12, `--threads=n` adds an
# interactive thread, which would make 65 threads for 64 exclusive cores.
# Expect about 15 minutes.

#SBATCH --partition=amdq
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1:00:00
#SBATCH --job-name=imexrk-stage-arithmetic
#SBATCH --output=imexrk-stage-arithmetic-%j.out

set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
REPO="${IMEXRK_REPO:-$PWD}"
cd "$REPO"
hostname
numactl -H | head -3
git log --oneline -1 || true
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

export IMEXRK_BENCH_THREADS=1,2,4,8,16,32,64
export IMEXRK_BENCH_BYTES=1e8

echo "=== pinned, first touch"
JULIA_EXCLUSIVE=1 IMEXRK_BENCH_INTERACTIVE=0 \
    srun --cpu-bind=none julia --project=. bench/stage_arithmetic.jl
echo "=== pinned, interleaved"
JULIA_EXCLUSIVE=1 IMEXRK_BENCH_INTERACTIVE=0 \
    srun --cpu-bind=none numactl --interleave=all julia --project=. bench/stage_arithmetic.jl
echo "=== unpinned, first touch"
srun --cpu-bind=none julia --project=. bench/stage_arithmetic.jl
