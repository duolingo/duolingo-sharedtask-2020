# this doesn't change.
src=en
tgt=vi

lang=${src}-${tgt}
DATA=data/courses/${lang}
DUMP=data/dumps/${lang}

# download these first.
# git clone https://github.com/moses-smt/mosesdecoder
# git clone https://github.com/rsennrich/subword-nmt
MOSES=$HOME/IdeaProjects/mosesdecoder
SUBWORDNMT=$HOME/IdeaProjects/subword-nmt

# used in run_pretrained.sh (don't change NBEST)
NBEST=10

CANDLIMIT=10
