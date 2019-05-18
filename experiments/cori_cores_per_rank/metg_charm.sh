#!/bin/bash
#SBATCH --account=m2294
#SBATCH --qos=regular
#SBATCH --constraint=haswell
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL

cores=$(( $(echo $SLURM_JOB_CPUS_PER_NODE | cut -d'(' -f 1) / 2 ))

module load craype-hugepages8M

function launch {
    cores_per_rank=$2
    ranks_per_node=$(( cores / cores_per_rank ))
    srun -n $(( $1 * ranks_per_node )) -N $1 --ntasks-per-node=$ranks_per_node --cpus-per-task=$(( cores_per_rank * 2 )) --cpu_bind cores ../../charm++/benchmark "${@:3}"
}

function repeat {
    local -n result=$1
    local n=$2
    result=()
    for i in $(seq 1 $n); do
        result+=("${@:3}")
        if (( i < n )); then
            result+=("-and")
        fi
    done
}

function sweep {
    for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18; do
        for rep in 0 1 2 3 4; do
            if [[ $rep -le $s ]]; then
                local args
                repeat args $3 -kernel compute_bound -iter $(( 1 << (26-s) )) -type $4 -radix ${RADIX:-5} -steps ${STEPS:-1000} -width $(( $2 * cores ))
                $1 $2 $5 "${args[@]}"
            fi
        done
    done
}

for n in $SLURM_JOB_NUM_NODES; do
    for g in ${NGRAPHS:-1}; do
        for t in ${PATTERN:-stencil_1d}; do
            for c in ${CORES_PER_RANK:-1}; do
                sweep launch $n $g $t $c > charm_ngraphs_${g}_type_${t}_coresperrank_${c}_nodes_${n}.log
            done
        done
    done
done