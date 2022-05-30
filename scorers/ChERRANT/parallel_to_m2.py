import os
from modules.annotator import Annotator
from modules.tokenizer import Tokenizer
import torch
import argparse
from collections import Counter
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool
from opencc import OpenCC

os.environ["TOKENIZERS_PARALLELISM"] = "false"

annotator, sentence_to_tokenized = None, None
cc = OpenCC("t2s")

def annotate(line):
    """
    :param line:
    :return:
    """
    sent_list = line.split("\t")[1:]
    source = sent_list[0]
    if args.segmented:
        source = source.strip()
    else:
        source = "".join(source.strip().split())
    output_str = ""
    for idx, target in enumerate(sent_list[1:]):
        try:
            if args.segmented:
                target = target.strip()
            else:
                target = "".join(target.strip().split())
            if not args.no_simplified:
                target = cc.convert(target)
            source_tokenized, target_tokenized = sentence_to_tokenized[source], sentence_to_tokenized[target]
            out, cors = annotator(source_tokenized, target_tokenized, idx)
            if idx == 0:
                output_str += "".join(out[:-1])
            else:
                output_str += "".join(out[1:-1])
        except Exception:
            raise Exception
    return output_str

def main(args):
    tokenizer = Tokenizer(args.granularity, args.device, args.segmented)
    global annotator, sentence_to_tokenized
    annotator = Annotator.create_default(args.granularity, args.multi_cheapest_strategy)
    lines = open(args.file, "r").read().strip().split("\n")  # format: id src tgt1 tgt2...
    # error_types = []

    with open(args.output, "w") as f:
        count = 0
        sentence_set = set()
        sentence_to_tokenized = {}
        for line in lines:
            sent_list = line.split("\t")[1:]
            for idx, sent in enumerate(sent_list):
                if args.segmented:
                    # print(sent)
                    sent = sent.strip()
                else:
                    sent = "".join(sent.split()).strip()
                if idx >= 1:
                    if not args.no_simplified:
                        sentence_set.add(cc.convert(sent))
                    else:
                        sentence_set.add(sent)
                else:
                    sentence_set.add(sent)
        batch = []
        for sent in tqdm(sentence_set):
            count += 1
            if sent:
                batch.append(sent)
            if count % args.batch_size == 0:
                results = tokenizer(batch)
                for s, r in zip(batch, results):
                    sentence_to_tokenized[s] = r  # Get tokenization map.
                batch = []
        if batch:
            results = tokenizer(batch)
            for s, r in zip(batch, results):
                sentence_to_tokenized[s] = r  # Get tokenization map.
        with Pool(args.worker_num) as pool:
            for ret in pool.imap(annotate, tqdm(lines), chunksize=8):
                if ret:
                    f.write(ret)
                    f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose input file to annotate")
    parser.add_argument("-f", "--file", type=str, required=True, help="Input parallel file")
    parser.add_argument("-o", "--output", type=str, help="Output file", required=True)
    parser.add_argument("-b", "--batch_size", type=int, help="The size of batch", default=128)
    parser.add_argument("-d", "--device", type=int, help="The ID of GPU", default=0)
    parser.add_argument("-w", "--worker_num", type=int, help="The number of workers", default=16)
    parser.add_argument("-g", "--granularity", type=str, help="Choose char-level or word-level evaluation", default="char")
    parser.add_argument("-m", "--merge", help="Whether merge continuous replacement/deletion/insertion", action="store_true")
    parser.add_argument("-s", "--multi_cheapest_strategy", type=str, choices=["first", "all"], default="all")
    parser.add_argument("--segmented", help="Whether tokens have been segmented", action="store_true")  # 支持提前token化，用空格隔开
    parser.add_argument("--no_simplified", help="Whether simplifying chinese", action="store_true")  # 将所有corrections转换为简体中文
    args = parser.parse_args()
    main(args)