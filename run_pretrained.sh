#!/bin/bash

set -e

# get the parent directory of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/variables.sh

# Set this to your favorite model!
MODEL=$1

OUTPUT=$DATA/output
mkdir -p $OUTPUT

fairseq-generate data/courses/en-${tgt}/bin/ --path $MODEL --dataset-impl raw --raw-text \
   --beam $NBEST --batch-size 128 --remove-bpe --nbest $NBEST --replace-unk > $OUTPUT/gen.out
 
# this cleans all the BPE
sed -i '' 's/@@ //g' $OUTPUT/gen.out

TESTFILE=${SHARED_TASK_DATA}/${src}_${tgt}/test.${src}_${tgt}.2020-01-13.gold.txt

python my_cands_extract.py --origfile $TESTFILE --infile $OUTPUT/gen.out --outfile $OUTPUT/all_cands.txt --candlimit $CANDLIMIT
mv sys.out ref.out ${OUTPUT}

cat ${OUTPUT}/all_cands.txt | $MOSES/scripts/tokenizer/detokenizer.perl -l $tgt > ${OUTPUT}/all_cands_detok.txt

# who creates all_cands.txt? my_cands_extract.py
python staple_2020_scorer.py --gold $TESTFILE --pred ${OUTPUT}/all_cands_detok.txt


# this will compute BLEU score (best translation vs the top hypothesis)
#fairseq-score --sys ${OUTPUT}/sys.out --ref ${OUTPUT}/ref.out
cat ${OUTPUT}/sys.out | $MOSES/scripts/tokenizer/detokenizer.perl > sys_detok
cat ${OUTPUT}/ref.out | $MOSES/scripts/tokenizer/detokenizer.perl > ref_detok

cat sys_detok | sacrebleu -lc ref_detok
rm sys_detok ref_detok
