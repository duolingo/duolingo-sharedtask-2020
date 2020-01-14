#!/bin/bash

set -e

# get the parent directory of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/variables.sh

SCRIPTS=$MOSES/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=$SUBWORDNMT/subword_nmt

### Make directories
tmp=$DATA/tmp
mkdir -p $DATA
mkdir -p $tmp

# Change this to the location where you downloaded the data.
SHARED_TASK_DATA=$HOME/staple2020/staple-2020-train

FOLDS=(train)

for fold in $FOLDS; do
    
    python get_traintest_data.py --fname ${SHARED_TASK_DATA}/${src}_${tgt}/${fold}.${src}_${tgt}.2020-01-13.gold.txt --srcfname $DATA/${fold}-sents.${src} --tgtfname $DATA/${fold}-sents.${tgt} --prefix ${fold}
done

echo "pre-processing train data..."
for fold in $FOLDS; do
    for l in $src $tgt; do
        tok=${fold}-sents.tok.${l}
        cat $DATA/${fold}-sents.${l} | perl $TOKENIZER -threads 8 -l $l | perl $LC > $tmp/${fold}-sents.clean.${l}
    done
done

# gather all the training data.
ALLTRAIN=$tmp/alltrain
rm -rf $ALLTRAIN
for l in $src $tgt; do
    cat $tmp/train-sents.clean.${l} >> $ALLTRAIN
done

# learn BPE
echo "learning bpe..."
BPE_CODE=${DATA}/bpecode
# data is pretty small, so keep this small.
BPE_TOKENS=20000
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $ALLTRAIN > $BPE_CODE

# Then apply bpe
outf=sents.clean.bpe
for fold in $FOLDS; do
    for l in $src $tgt; do
        f=${fold}-sents.clean.${l}
        echo "Applying BPE to ${f}..."
        cat $tmp/$f | python $BPEROOT/apply_bpe.py -c $BPE_CODE > ${DATA}/${fold}-${outf}.${l}
    done
done
rm -rf $tmp

echo "Files are in ${DATA}/${outf}.{$src, $tgt}"

##########################################################################################

rm -rf ${DATA}/bin
mkdir ${DATA}/bin

# Since we only provide training data, you may want to make your own split.
fairseq-preprocess --source-lang $src --target-lang $tgt \
    --trainpref $DATA/train-${outf} --validpref $DATA/dev-${outf} --testpref $DATA/test-${outf} \
    --destdir ${DATA}/bin/ --workers 20 --dataset-impl raw
