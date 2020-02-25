import argparse
from collections import defaultdict

from utils import FIELDSEP, read_trans_prompts


def main(origfile: str, infile: str, outfile: str, candlimit: int):
    """
    This processes the output of fairseq-generate so that it can be scored with sacrebleu and 
    so that it has the shared task format. 

    fairseq-generate does this annoying thing where the order of the outputs is not guaranteed, but 
    it retains order information in the metadata. So we just need to keep track of that and use it to
    order the outputs correctly.

    """

    sysfile = "sys.out"
    reffile = "ref.out"

    with open(origfile) as f:
        orig_prompts = read_trans_prompts(f.readlines())

    # this variable is used to write out only the top hypothesis for scoring bleu
    first = True
    cands = 0

    # maps: order from metadata to list of lines to be written to file. 
    # make sure to sort this by key before writing to file.
    outd = defaultdict(list)
    refd = defaultdict(list)
    sysd = defaultdict(list)

    with open(infile) as f:
        for line in f:
            sline = line.strip().split("\t")
            if line.startswith("S-"):
                num = int(sline[0].split("-")[-1])
                outd[num].append(sline[1])
                first = True
                cands = 0
            elif line.startswith("T-"):
                num = int(sline[0].split("-")[-1])
                # this is the reference
                refd[num].append(sline[1] + "\n")
            elif line.startswith("H-"):
                num = int(sline[0].split("-")[-1])
                # this is the prediction, there may be many of these.
                if candlimit == -1 or cands < candlimit:
                    outd[num].append(sline[2] + "\n")
                    cands += 1

                # only write the first of these.
                if first:
                    sysd[num].append(sline[2] + "\n")
                    first = False
            else:
                pass

    with open(outfile, "w") as out, open(sysfile, "w") as sf, open(reffile, "w") as rf:    
        # orig_prompts has a particular order, and when we sort outd by 
        # the items, the order should match.
        for orig_prompt,item in zip(orig_prompts, sorted(outd.items())):
            num,linelist = item
            firstline = linelist[0]
            textID = orig_prompt[0]
            out.write(f"\n{textID}{FIELDSEP}{firstline}\n")

            for line in linelist[1:]:
                out.write(line)

        for num,linelist in sorted(sysd.items()):
            for line in linelist:
                sf.write(line)

        for num,linelist in sorted(refd.items()):
            for line in linelist:
                rf.write(line)

    print(f"Wrote to {outfile}, {sysfile}, {reffile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("This processes the output of fairseq-generate so that it can be scored with sacrebleu and so that it has the shared task format. ")
    parser.add_argument("--origfile", help="Name of the original shared task file, with original prompt IDs.", required=True)
    parser.add_argument("--infile", help="Name of output file from fairseq-generate, probably called gen.out", required=True)
    parser.add_argument("--outfile", help="Name of desired output file. This will be the shared task format file.", required=True)
    parser.add_argument("--candlimit", help="Max number of candidates to put in file (default is -1, meaning all)", type=int, default=-1)
    args = parser.parse_args()

    main(args.origfile, args.infile, args.outfile, args.candlimit)
