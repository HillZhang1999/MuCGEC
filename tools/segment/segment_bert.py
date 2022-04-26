import sys
import tokenization
from tqdm import tqdm
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(dir_path, "vocab.txt"), do_lower_case=False)

for line in tqdm(sys.stdin):
    line = line.strip()
    origin_line = line
    line = line.replace(" ", "")
    line = tokenization.convert_to_unicode(line)
    if not line:
        continue
    tokens = tokenizer.tokenize(line)
    print(' '.join(tokens))