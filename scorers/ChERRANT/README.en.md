# Instruction

We have built a Chinese GEC assessment tool, ChERRANT (Chinese ERRANT), using the mainstream English GEC assessment tool [ERRANT](https://github.com/chrisjbryant/errant) for reference. The main function of ChERRANT is to calculate the accuracy, recall and f-value of prediction results by comparing prediction editing and standard editing, thereby evaluating the performance of syntax error correction model.

## Environment

`Requirements Txt` contains the main environment required for the experiment. The specific environment construction process is as follows:

```
conda create -n cherrant python==3.8
conda activate cherrant
pip install -r requirements.txt
```

## Usage

### Overview

#### File Format

The format of the predicted file is: `id \t Original sentence \t Forecast results `;

The format of the standard answer file is: `id \t Original sentence \t Standard answer 1 \t Standard answer 2 \t ... `;

The format of the edited file is m2, as follows:

```
S 冬 阴 功 是 泰 国 最 著 名 的 菜 之 一 ， 它 虽 然 不 是 很 豪 华 ， 但 它 的 味 确 实 让 人 上 瘾 ， 做 法 也 不 难 、 不 复 杂 。
T0-A0 冬 阴 功 是 泰 国 最 著 名 的 菜 之 一 ， 它 虽 然 不 是 很 豪 华 ， 但 它 的 味 道 确 实 让 人 上 瘾 ， 做 法 也 不 难 、 不 复 杂 。
A 27 27|||M|||道|||REQUIRED|||-NONE-|||0
```

+ `S` represents the original sentence;
+ `T0-A0` represents the 0th editing sequence of the 0th answer (a sentence may have multiple answers, and an answer may also have multiple editing sequences with the same editing distance);
+ `A  ` represents editing, mainly including the following information: The start and end positions of errors (`27 27`); Error type (`M`, Missing Error); Modification method of errors (`道`, i.e. insert "道"); Annotation ID (`0`);

#### Evaluation Process

The main evaluation steps are as follows:

1. Converting parallel file of standard answers to edit file `gold.m2` through `parallel_to_m2.py`(it is only necessary for the first evaluation and can be reused later);
2. Converting parallel file of predicted answers to edit file `hyp.m2` through `parallel_to_m2.py`;
3. Comparing `gold.m2` and `hyp.m2` using `compare_m2_for_evaluation.py` to get the final evaluation indicators;

For example scripts of the complete process, please refer to `./demo.sh`.

### Extract Edit

First, merging the input file (one sentence per line) and the output file (one sentence per line) into a parallel format:

```
INPUT_FILE=./samples/demo.input
OUTPUT_FILE=./samples/demo.hyp
HYP_PARA_FILE=./samples/demo.hyp.para

paste $INPUT_FILE $OUTPUT_FILE | awk '{print NR"\t"$p}' > $HYP_PARA_FILE
```

Then, extracting the edit with the following command:

```
HYP_M2_FILE=./samples/demo.hyp.m2.char

python parallel_to_m2.py -f $HYP_PARA_FILE -o $HYP_M2_FILE -g char
```

By default, word level edit is extracted.

By setting the ` -g 'parameter to' word ', you can extract word level edit. Although word level edit has marked more error type information, it may be affected by word segmentation errors. For more settings, refer to the command line help file:

```
python parallel_to_m2.py --help
```

### Calculate Evaluation

Using the following script to compare the predicted edit file with the standard edit file to get the word level evaluation indicators:

```
REF_M2_FILE=./samples/demo.ref.m2.char
python compare_m2_for_evaluation.py -hyp $HYP_M2_FILE -ref $REF_M2_FILE
```

The F0.5 index at the word level is the official evaluation index adopted by the MuCGEC data set, and the evaluation results are as follows:

```
=========== Span-Based Correction ============
TP      FP      FN      Prec    Rec     F0.5
8       19      35      0.2963  0.186   0.2649
==============================================
```

The program can also support more fine-grained information display, such as display of detection indicators and correction indicators of different types of errors. If necessary, please refer to the command line help information for use.

```
python compare_m2_for_evaluation.py --help
```

## Citation

If you find this work is useful for your research, please cite our paper:

#### MuCGEC: a Multi-Reference Multi-Source Evaluation Dataset for Chinese Grammatical Error Correction (Accepted by NAACL2022 main conference) [[PDF]](https://arxiv.org/pdf/2204.10994.pdf)

```
@inproceedings{zhang-etal-2022-mucgec,
    title = "{MuCGEC}: a Multi-Reference Multi-Source Evaluation Dataset for Chinese Grammatical Error Correction",
    author = "Zhang, Yue and Li, Zhenghua and Bao, Zuyi and Li, Jiacheng and Zhang, Bo and Li, Chen and Huang, Fei and Zhang, Min",
    booktitle = "Proceedings of NAACL-HLT",
    year = "2022",
    address = "Online",
    publisher = "Association for Computational Linguistics"
```
