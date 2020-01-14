#!/bin/bash

set -e

# get the parent directory of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/variables.sh

COURSE=${src}-${tgt}
DATA_DIR=data/courses/$COURSE/bin/

CP_DIR=checkpoints/fconv/${src}-${tgt}
mkdir -p $CP_DIR

fairseq-train \
     $DATA_DIR \
     --arch fconv \
     --max-tokens 8192 --save-dir $CP_DIR --keep-last-epochs 2 --dataset-impl raw \
     --optimizer adam --adam-betas '(0.9, 0.98)' \
     --lr 5e-4 \
     --clip-norm 0.0 \
     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
     --lr-scheduler inverse_sqrt --warmup-updates 4000 \
     --dropout 0.3 --weight-decay 0.0001
