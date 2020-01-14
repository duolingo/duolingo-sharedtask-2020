import argparse
import hashlib

from sacremoses import MosesDetokenizer

from utils import FIELDSEP, makeID


def main(infile: str, outfile: str, candlimit: int):
    """
    This processes the output of fairseq-generate so that it can be scored with sacrebleu and 
    so that it has the shared task format. 
    """

    sysfile = "sys.out"
    reffile = "ref.out"

    md = MosesDetokenizer(lang="en")

    with open(infile) as f, open(outfile, "w") as out, open(sysfile, "w") as sf, open(reffile, "w") as rf:
        first = True
        cands = 0
        for line in f:
            sline = line.strip().split("\t")
            if line.startswith("S-"):
                # it's hard to have fairseq pass prompt ids through the training/evaluation process
                # so we resort to regenerating ids based on the prompt text.
                # we have to be careful that the text is *exactly* the same, or the id generation will be wrong.
                sline_detok = md.detokenize(sline[1].split(" "))
                textID = makeID(sline_detok)
                out.write(f"\n{textID}{FIELDSEP}{sline[1]}\n")
                first = True
                cands = 0
            elif line.startswith("T-"):
                # this is the reference
                rf.write(sline[1] + "\n")
            elif line.startswith("H-"):
                # this is the prediction, there may be many of these.
                if candlimit == -1 or cands < candlimit:
                    out.write(sline[2] + "\n")
                    cands += 1

                # only write the first of these.
                if first:
                    sf.write(sline[2] + "\n")
                    first = False
            else:
                pass

    print(f"Wrote to {outfile}, {sysfile}, {reffile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("This processes the output of fairseq-generate so that it can be scored with sacrebleu and so that it has the shared task format. ")
    parser.add_argument("--infile", help="Name of output file from fairseq-generate, probably called gen.out", required=True)
    parser.add_argument("--outfile", help="Name of desired output file. This will be the shared task format file.", required=True)
    parser.add_argument("--candlimit", help="Max number of candidates to put in file (default is -1, meaning all)", type=int, default=-1)
    args = parser.parse_args()

    main(args.infile, args.outfile, args.candlimit)
