import argparse

from utils import read_trans_prompts, read_transfile


def get_data(fname: str, srcfname: str, tgtfname: str, prefix: str) -> None:
    """
    This converts data in the shared task format into standard machine translation format (one sentence per line, languages in separate files.)
    For training data, it combines the prompt with all accepted translations. 
    For dev or test data, it combines the prompt only with the most popular translation.
    """

    with open(fname) as f:
        lines = f.readlines()
    d = read_transfile(lines, strip_punc=False, weighted=True)
    id_text = dict(read_trans_prompts(lines))

    with open(srcfname, "w") as src, open(tgtfname, "w") as tgt:
        for idstring in d.keys():

            # prompt is combination of id and text.
            prompt = id_text[idstring]
            ats = d[idstring]

            # make sure that the first element is the largest.
            ats = sorted(ats.items(), key=lambda p: p[1], reverse=True)

            # if it is train
            if prefix == "train":
                # write all pairs.
                for p in ats:
                    print(prompt, file=src)
                    print(p[0], file=tgt)
            else:
                # write just the first pair (evaluate only on first line.)
                top_ranked_text = ats[0][0]
                print(prompt, file=src)
                print(top_ranked_text, file=tgt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("This converts data in the shared task format into standard machine translation format (one sentence per line, languages in separate files.)")
    parser.add_argument("--fname", help="Path of shared task file (probably something like train.en_vi.2020-01-13.gold.txt)", required=True)
    parser.add_argument("--srcfname", help="Name of desired src file, probably something like train_sents.en", required=True)
    parser.add_argument("--tgtfname", help="Name of desired tgt file, probably something like train_sents.vi", required=True)
    parser.add_argument("--prefix", help="One of [train, dev, test]", choices=["train", "dev", "test"])
    args = parser.parse_args()

    get_data(args.fname, args.srcfname, args.tgtfname, args.prefix)
