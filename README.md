# STAPLE 2020 README

Welcome to the [Duolingo](https://www.duolingo.com/) 2020 Shared Task! The shared task website is here: [sharedtask.duolingo.com](http://sharedtask.duolingo.com/).

This repository has code for:

* Scoring a predictions file
* Training an example baseline model with [fairseq](https://github.com/pytorch/fairseq)

Python 3.6+ is required. It is strongly recommended that you run this in a virtual environment.



## Scoring

### Requirements

There are no special requirements for running the scoring function.

### Code

You can score a predicted file as follows (using the AWS baseline as example, and running in the repo top level directory):

```bash
$ python staple_2020_scorer.py --goldfile staple-2020-train/en_vi/train.en_vi.2020-01-13.gold.txt  --predfile staple-2020-train/en_vi/train.en_vi.aws_baseline.pred.txt
```


## Training models

If all you want to do is evaluation, then ignore this section.

Most participants will probably write their own code for this task, but we also provide code for training a vanilla
sequence-to-sequence models using fairseq. This does not produce the best results for this task,
but it is an obvious baseline and may give you a jumpstart. This code is an adaptation of [translation tutorials](https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md) from fairseq.

### Requirements

Certain scripts require perl to run. If you are on mac or Linux, you probably already have it. See [here](https://www.perl.org/get.html) for more details.

Next, get these repositories:
```bash
$ git clone https://github.com/moses-smt/mosesdecoder
$ git clone https://github.com/rsennrich/subword-nmt
```

Go to the `variables.sh` file and set the paths for `MOSES` and `SUBWORDNMT` accordingly.

Install python requirements:

```bash
$ pip install fairseq sacremoses subword_nmt sacrebleu tqdm
```

### Code

The following files are provided.

* `variables.sh` : common BASH variables
* `preprocess.sh` : to preprocess the data for training with fairseq
* `train.sh` : to train the model using preprocessed data
* `run_pretrained.sh` : script to run pretrained fairseq models
* `my_cands_extract.py` : used to convert outputs from fairseq into shared task format files (used in `run_pretrained.sh`).
* `get_traintest_data.py` : converts shared task format files into fairseq-readable format (used in `preprocess.sh`).

The most relevant files are `preprocess.sh`, `train.sh`, and `run_pretrained.sh`.

Good luck!

If you have questions, feel free to check or post to the [mailing list](https://groups.google.com/forum/#!forum/duolingo-sharedtask-2020)
