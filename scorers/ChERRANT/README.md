# 使用说明

我们借鉴了英文上主流的GEC评估工具[ERRANT](https://github.com/chrisjbryant/errant)，搭建了中文GEC评估工具ChERRANT（Chinese ERRANT）。ChERRANT的主要功能是通过对比预测编辑和标准编辑，计算预测结果的精确度、召回度、F值指标，从而评估语法纠错模型的性能。

## 环境

`requirements.txt`包含了实验所需的主要环境，具体环境搭建流程如下所示：

```
conda create -n cherrant python==3.8
conda activate cherrant
pip install -r requirements.txt
```

## 使用方式

### 总览

#### 文件格式

预测文件格式为：`id \t 原句 \t 预测结果 `；

标准答案文件格式为：`id \t 原句 \t 标准答案1 \t 标准答案2 \t ... `；

编辑文件格式采用M2格式，如下所示：

```
S 冬 阴 功 是 泰 国 最 著 名 的 菜 之 一 ， 它 虽 然 不 是 很 豪 华 ， 但 它 的 味 确 实 让 人 上 瘾 ， 做 法 也 不 难 、 不 复 杂 。
T0-A0 冬 阴 功 是 泰 国 最 著 名 的 菜 之 一 ， 它 虽 然 不 是 很 豪 华 ， 但 它 的 味 道 确 实 让 人 上 瘾 ， 做 法 也 不 难 、 不 复 杂 。
A 27 27|||M|||道|||REQUIRED|||-NONE-|||0
```

+ `S` 代表原句；
+ `T0-A0`代表第0个答案的第0个编辑序列（一个句子可能有多个答案，一个答案也可能有编辑距离相同的多个编辑序列）；
+ `A  `代表编辑，主要包括如下信息：错误的起始和结束位置（`27 27`）；错误类型（`M`，Missing Error，缺失错误）；错误的修改答案（`道`，即插入“道”）；标注id（`0`）；

#### 评估过程

主要的评估步骤为：

1. 将标准答案平行文件通过`parallel_to_m2.py`转换成M2格式的编辑文件`gold.m2`（仅首次评估需要，之后可以复用）；
2. 将预测答案平行文件通过`parallel_to_m2.py`转换成M2格式的编辑文件`hyp.m2`；
3. 使用`compare_m2_for_evaluation.py`对比`hyp.m2`和`gold.m2`，得到最终的评价指标。

完整流程的示例脚本可以参考`./demo.sh`。

### 抽取编辑

首先，将输入文件（每行一句）和输出文件（每行一句）合并，处理成平行格式：

```
INPUT_FILE=./samples/demo.input
OUTPUT_FILE=./samples/demo.hyp
HYP_PARA_FILE=./samples/demo.hyp.para

paste $INPUT_FILE $OUTPUT_FILE | awk '{print NR"\t"$p}' > $HYP_PARA_FILE
```

然后，通过如下命令抽取编辑：

```
HYP_M2_FILE=./samples/demo.hyp.m2.char

python parallel_to_m2.py -f $HYP_PARA_FILE -o $HYP_M2_FILE -g char
```

默认抽取的是字级别编辑。

通过将`-g`参数设为`word`，则可以抽取词级别编辑。虽然词级别编辑标注了更多的错误类型信息，但可能会受到分词错误影响，因此仅供参考。更多设置请参考命令行帮助文件：

```
python parallel_to_m2.py --help
```

### 计算指标

使用如下脚本对比预测编辑文件和标准编辑文件，即可得到字级别的评测指标：

```
REF_M2_FILE=./samples/demo.ref.m2.char
python compare_m2_for_evaluation.py -hyp $HYP_M2_FILE -ref $REF_M2_FILE
```

字级别的F0.5指标是MuCGEC数据集采用的官方评测指标，评价结果如下所示：

```
=========== Span-Based Correction ============
TP      FP      FN      Prec    Rec     F0.5
8       19      35      0.2963  0.186   0.2649
==============================================
```

该程序也可支持更细粒度的信息展示，如展示检测指标和不同类型错误的纠正指标，请参考命令行帮助信息使用。

```
python compare_m2_for_evaluation.py --help
```

## 引用

如果您使用了我们的评价程序，请引用我们的论文：

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
