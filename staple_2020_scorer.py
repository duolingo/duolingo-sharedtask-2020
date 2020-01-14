import argparse
import string
from typing import Dict, List, Set

from utils import read_transfile

def score(gold: Dict[str, Dict[str, float]], pred: Dict[str, Dict[str, float]], verbose: bool=False) -> float:

    # now we score gold and pred!
    gold_keys = set(gold.keys())
    pred_keys = set(pred.keys())

    print(f"There are {len(gold_keys)} keys in gold and {len(pred_keys)} keys in pred.")

    if len(gold_keys.intersection(pred_keys)) == len(pred_keys):
        print("All predicted keys are in the gold. WELL DONE.")

    extras = pred_keys - gold_keys
    if len(extras) > 0:
        print(f"Warning: your pred file has {len(extras)} key sentences that don't appear in the gold.")
        print(sorted(list(extras)))
        print("unmatched golds:")
        unmatched_golds = sorted(list(gold_keys - pred_keys))
        print(unmatched_golds)
                
    if len(gold.keys()) != len(pred.keys()):
        print(f"WARNING: num keys doesn't match: {len(gold.keys())}, {len(pred.keys())}")
        print("missing keys:", gold_keys - pred_keys)

    sent_wf1s = []
    sent_f1s = []
    micro_tp = 0
    micro_wtp = 0
    micro_fp = 0
    micro_fn = 0
    micro_wfn= 0
    for k, gold_opts in gold.items():
        # gold_opts is now a dictionary of {at: weight}
        if k not in pred:
            print(f"A gold key ({k}) is not in the predicted file. This should never happen!")
        else:
            pred_opts = pred[k]
            sentences_only_in_gold = gold_opts.keys() - pred_opts.keys()
            fn = len(sentences_only_in_gold)
            # 'w' character stands for weighted
            wfn = sum([gold_opts[o] for o in sentences_only_in_gold])

            sentences_in_both = set(gold_opts.keys()).intersection(set(pred_opts.keys()))
            tp = len(sentences_in_both)
            wtp = sum([gold_opts[o] for o in sentences_in_both])

            # intentionally no wfp
            sentences_only_in_pred = pred_opts.keys() - gold_opts.keys()
            fp = len(sentences_only_in_pred)

            micro_tp += tp
            micro_wtp += wtp
            micro_fp += fp
            micro_fn += fn
            micro_wfn += wfn

            if verbose:
                print(k)
                print("true positives") 
                for i in sorted([(o,gold_opts[o]) for o in sentences_in_both], key=lambda p: p[1], reverse=True):
                    print(i)
                print("false positives", sentences_only_in_pred)
                print("false negatives")
                for i in sorted([(o,gold_opts[o]) for o in sentences_only_in_gold], key=lambda p: p[1], reverse=True):
                    print(i)
                print()

        # calculate MACRO
        
        precision = 0 if tp+fp == 0 else tp / (tp + fp)
        recall = 0 if tp+fn == 0 else tp / (tp + fn)
        weighted_recall = 0 if wtp+wfn == 0 else wtp / (wtp + wfn)

        if precision == 0 and recall == 0:
            macro_f1 = 0
        else:
            macro_f1 = 2*precision*recall / (precision + recall)

        if precision == 0 and weighted_recall == 0:
            macro_weighted_f1 = 0
        else:
            macro_weighted_f1 = 2*precision*weighted_recall / (precision + weighted_recall)

        sent_f1s.append(macro_f1)
        sent_wf1s.append(macro_weighted_f1)

    precision = 0 if micro_tp + micro_fp == 0 else micro_tp / (micro_tp + micro_fp)
    recall = 0 if micro_tp + micro_fn == 0 else micro_tp / (micro_tp + micro_fn)
    weighted_recall = 0 if micro_wtp + micro_wfn == 0 else micro_wtp / (micro_wtp + micro_wfn)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2*precision*recall / (precision + recall)
    if precision + weighted_recall == 0:
        weighted_f1 = 0
    else:
        weighted_f1 = 2*precision*weighted_recall / (precision + weighted_recall)

    macro_weighted_f1 = sum(sent_wf1s) / float(len(sent_wf1s))
    macro_f1 = sum(sent_f1s)/float(len(sent_f1s))

    print(f"Precision:         {precision:.2%}")
    print(f"Recall:            {recall:.2%}")
    print(f"Weighted Recall:   {weighted_recall:.2%}")
    print(f"Micro F1:          {f1:.2%}")
    print(f"Macro F1:          {macro_f1:.2%}")
    print(f"Weighted Micro F1: {weighted_f1:.2%}")
    print(f"Weighted Macro F1: {macro_weighted_f1:.2%}")

    # This may be helpful for reporting scores in e.g. spreadsheets or latex tables.
    # print(precision, recall, weighted_recall, f1, macro_f1, weighted_f1, macro_weighted_f1)

    return macro_weighted_f1


if __name__ == "__main__":
    # this will take two files of the format:
    # id|source
    # trans1
    # trans2
    # trans3
    # ...
    # trans4
    #
    # id|source
    # trans1
    # ...

    # and it will score them with F1. 
    # to be precise: 
    # sent_i is associated with a set of "gold" translations {trans_j, ...}, which may or may not have some weights associated with them.
    # for sent_i, we predict a set of translations {trans_k, ...}. These are scored against the gold set as follows:

    # tp : | intersection of the sets |
    # fn : | gold - pred |
    # fp : | pred - gold |

    # this outputs both micro f1, in which precision/recall/f1 are calculated over the entire dataset, and macro f1, in which precision/recall/f1
    # are calculated over each prompt separately and averaged at the end.

    parser = argparse.ArgumentParser()
    parser.add_argument("--goldfile", help="gold file", required=True)
    parser.add_argument("--predfile", help="pred file", required=True)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    args = parser.parse_args()

    with open(args.goldfile) as f:
        print("reading gold")
        gold = read_transfile(f.readlines(), weighted=True)

    with open(args.predfile) as f:
        print("reading pred")
        pred = read_transfile(f.readlines())

    score(gold, pred, args.verbose)
