#!/bin/bash

RUNDS="StanfordDogs,StanfordCars,Caltech101,OxfordFlowers"

ATTEMPT=$1
SEED=$2
RUNDS=$3
ARCH=$4
MULTICL=$5

DIR=results/results${ATTEMPT}/seed_${SEED}/
if [ -e "${DIR}results_${ds}.pkl" ]; then
    echo "Oops! The results exist at '${DIR}results_${ds}.pkl' (so skip this job)"
else
    echo "Saving dir ${DIR}"
    python3 main.py \
    --output_dir results/results${ATTEMPT}/seed_${SEED}/ \
    --seed ${SEED} \
    --run_ds ${RUNDS} \
    --backbone_arch ${ARCH} \
    --multiclass_forget ${MULTICL} 
fi
