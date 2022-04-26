import sys
import os
import tokenization
from tqdm import tqdm
from segger.pkunlp import Segmentor

file_path = os.path.dirname(os.path.abspath(__file__))
tokenizer = Segmentor(os.path.join(file_path, "segger/feature/segment.feat"), os.path.join(file_path, "segger/feature/segment.dic"))

with open(sys.argv[1], "w", encoding="utf-8") as f:
    for line in tqdm(sys.stdin):
        line = "".join(line.strip().split())
        origin_line = line
        if not line:
            continue
        tokens = tokenizer.seg_string(line)
        # print(" ".join(tokens))
        f.write(' '.join(tokens) + "\n")
