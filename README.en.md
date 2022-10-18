# MuCGEC: A Multi-Reference Multi-Source Evaluation Dataset for Chinese Grammatical Error Correction & SOTA Models


English | [简体中文](README.md)

## News
+ Our new work SynGEC has been accepted by EMNLP2022. In this paper, we propose a SynGEC model that can utilize GEC-oriented tailored syntax and get SOTA performance (e.g., 45.32/46.51 F0.5 scores on NLPCC-18 and MuCGEC-Test), respectively. Welcome to test it! Link: [[Link]](https://github.com/HillZhang1999/SynGEC) 

+ We have uploaded the full MuCGEC dataset to this [link](https://github.com/HillZhang1999/MuCGEC/tree/main/data/MuCGEC) and opened a leaderboard at [Tianchi Platform](https://tianchi.aliyun.com/dataset/dataDetail?dataId=131328#4), welcome to submit your results!

+ We have released a part of MuCGEC dataset at [Tianchi Platform](https://tianchi.aliyun.com/dataset/dataDetail?dataId=131328), welcome to download and test!

## Citation
If you find this work is useful for your research, please cite our paper:

#### MuCGEC: a Multi-Reference Multi-Source Evaluation Dataset for Chinese Grammatical Error Correction (Accepted by NAACL2022 main conference) [[PDF]](https://aclanthology.org/2022.naacl-main.227.pdf)

```
@inproceedings{zhang-etal-2022-mucgec,
    title = "{M}u{CGEC}: a Multi-Reference Multi-Source Evaluation Dataset for {C}hinese Grammatical Error Correction",
    author = "Zhang, Yue  and
      Li, Zhenghua  and
      Bao, Zuyi  and
      Li, Jiacheng  and
      Zhang, Bo  and
      Li, Chen  and
      Huang, Fei  and
      Zhang, Min",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.227",
    pages = "3118--3130",
    }
```

## Introduction

Chinese Grammatical Error Correction (CGEC) technology aims to automatically correct spelling, grammatical, semantical, and other errors in a Chinese sentence. This technology is very useful in various scenarios.

Current CGEC evaluation datasets have some flaws, such as small amounts of data, single reference, single text source, etc. In order to better evaluate CGEC models, this repository provide a new multi-reference multi-source CGEC evaluation dataset named **MuCGEC**. Meanwhile, in order to promote the development of CGEC, we also provide the following additional resources:

+ **CGEC data annotation guidelines`./guidelines`**

+ **CGEC evaluation tools`./scorers`**:
  + `ChERRANT`：We extend the English GEC evaluation tool [ERRANT](https://github.com/chrisjbryant/errant) to accomondate Chinese, and name it `ChERRANT`(**Ch**inese **ERRANT**). ChERRANT supports the CGEC model evaluation at both word and character granularity.

+ **CGEC benchmark models`./models`**:
  + **Seq2Edit model`./models/seq2edit-based-CGEC`**: This kind of models treats GEC as a sequence labeling task and performs error corrections via a sequence of token-level edits, including insertion, deletion, and substitution.
    +  With minor modifications to accommodate Chinese, we adopt [GECToR](https://github.com/grammarly/gector), which achieves the SOTA performance on English GEC datasets.
  + **Seq2Seq model`./models/seq2seq-based-CGEC`**：This kind of models straightforwardly treats GEC as a monolingual translation task
    + We fine-tune the recently-proposed Seq2Seq pretrained model [Chinese-BART](https://github.com/fastnlp/CPT) and use it in CGEC.
  + **Ensemble model`./scorers/ChERRANT/emsemble.sh`**：We adopt a simple edit-wise vote mechanism, which can support the ensemble of heterogeneous models (such as Seq2Seq and Seq2Edit) and lead to significant performance boosts.
+ **CGEC tools`./tools`**：
  + **Tokenization tools**
  + **Data augmentation tools** (*Todo*)
  + **Data cleaning tools** (*Todo*)

## MuCGEC Dataset

Our dataset consists of texts written by Chinese learners. We select data from the following three sources: `NLPCC18` corpus, `CGED` corpus, and Chinese `Lang8` corpus. Each sentence has been corrected by three
annotators, and their corrections are meticulously reviewed by an expert, resulting in 2.3 references per sentence. The detailed statistics are shown below:

| Subset | #Sents | %Errors | Chars/sent | Edits/sent | Refs/sent |
| :------- | :---------: | :---------: | :---------: | :---------: |  :---------: |
| **MuCGEC-NLPCC18** | 1996 | 1904(95.4%) | 29.7 | 2.5 | 2.5 |
| **MuCGEC-CGED** | 3125 | 2988(95.6%) | 44.8 | 4.0 | 2.3 |
| **MuCGEC-Lang8** | 1942 | 1652(85.1%) | 37.5 | 2.8 | 2.1 |
| **MuCGEC-ALL** | 7063 | 6544(92.7%) | 38.5 | 3.2 | 2.3 |

Compared with previous CGEC evaluation sets (such as NLPCC18-orig and CGED-orig), MuCGEC has richer references and data sources.  In addition, during the annotation procedure, we also found that 74 sentences could not be annotated since their meanings are unclear.

For more details about MuCGEC, please kindly refer to our paper.

### Download link
We have released MuCGEC at [https://tianchi.aliyun.com/dataset?spm=5176.12281949.J_3941670930.15.493e2448TJV71D](https://tianchi.aliyun.com/dataset?spm=5176.12281949.J_3941670930.15.493e2448TJV71D).
<!-- **Note: we are currently planning a CGEC evaluation task at CCL2022 conference based on MuCGEC, so MuCGEC has not been released for the time being. We will release it soon. Please wait patiently.** -->

## CGEC benchmark models

### Experimental enviroment

We use Python 3.8 to experiment, and the necessary dependencies can be installed through the following code. Considering that there are some conflicts between the environments of Seq2Seq and Seq2Edit models, two environments need to be installed separately:
```
# Seq2Edit environment
pip install -r requirements_seq2edit.txt

# Seq2Seq environment
pip install -r requirements_seq2seq.txt
```

### Training data

The training data used in our experiment is composed of: 1) Chinese `Lang8` corpus; 2)`HSK` corpus. We upsampling `HSK` corpus 5 times. We only use the erroneous part of the training data.

Download Link: [Google Drive](https://drive.google.com/file/d/1l0A50z7fMXjQT3y2ct7TQsEHqOwlvg0_/view?usp=sharing)

<!-- **Note: the copyright issue about HSK data is under discussion, so the training data download link is not available at present.** -->

### Usage
We provide pipeline scripts to use our model, including the process of preprocessing->training->inference. Please refer to
`./models/seq2edit-based-CGEC/pipeline.sh` and `./models/seq2seq-based-CGEC/pipeline.sh`

Besides, we also provide converged checkpoints for testing (the following metrics are precision/recall/F0.5):

| Model | NLPCC18-Official(m2socrer)| MuCGEC(ChERRANT)|
| :-------: | :---------:| :---------: |
| **seq2seq_lang8**[[Link](https://drive.google.com/file/d/1Jras2Km4ScdVB0sx8ePg-PqCmDC4O8v5/view?usp=sharing)] |     37.78/29.91/35.89      | 40.44/26.71/36.67 |
| **seq2seq_lang8+hsk**[[Link](https://drive.google.com/file/d/180CXiW7pDz0wcbeTgszVoBrvzRmXzeZ9/view?usp=sharing)] |     41.50/32.87/39.43      | 44.02/28.51/39.70 |
| **seq2edit_lang8**[[Link](https://drive.google.com/file/d/13OAJ9DSThqssl93bSn0vQetetLhQz5LA/view?usp=sharing)] |     37.43/26.29/34.50      | 38.08/22.90/33.62 |
| **seq2edit_lang8+hsk**[[Link](https://drive.google.com/file/d/1ce7t8r3lLUeJ4eIxIg3EpXwHIUE99nk8/view?usp=sharing)] |     43.12/30.18/39.72      | 44.65/27.32/39.62 |

The ensemble strategy used in our paper can be found in `./scorers/ChERRANT/ensemble.sh`. Please kindly note that above seq2edit checkpoints are based on `Chinese-StructBERT-large` and seq2seq checkpoints are based on `Chinese-BART-Large`.

### Tips
+ We found that some useful tricks in English are still effective in Chinese, such as the confidence bias trick in GECToR and R2L reranking trick in Seq2Seq models. You can try them yourself.
+ We found that the decomposition of the training procedure (firstly train on HSL+Lang8, then fine-tune on HSK) can lead to further improvement. You can re-train our models following this two-stage training strategy.
+ Our Seq2Seq models based on [Chinese-BART](https://huggingface.co/fnlp/bart-large-chinese) can be improved: 1) the original vocabulary of Chinese-BART lacks some common Chinese punctuation / characters; 2）The training and generation speed of transformers library is relatively slow. We recently re-implement our Seq2Seq model based on [fairseq](https://github.com/pytorch/fairseq) and use some additional tricks, which greatly improved its performance (4-5 F0.5) and accelerated its speed. We will also release the improved version in the future.
+ None of the baselines we provided used pseudo data. For data augmentation, please refer to our paper in CTC-2021 CGEC competition [[Link]](https://github.com/HillZhang1999/CTC-Report). 

### Evaluation

To get the evaluation results on the [Official NLPCC18 dataset](http://tcci.ccf.org.cn/conference/2018/taskdata.php), you can make predictions by using our benchmark models, and evaluate the results through [M2Scorer](https://github.com/nusnlp/m2scorer). It is important to note that the predicted results must be segmented using the PKUSEG tool.

To get the evaluation results on MuCGEC, you can evaluate the results through our proposed [ChERRANT](./scorers/ChERRANT). Please refer to `./scorers/ChERRANT/demo.sh` for the use of ChERRANT.

**Error types in ChERRANT**
+ Operation tier (word/char granularity):
  + M(missing): Missing error, which means tokens need to be inserted;
  + R(redundant): Redundant error, which means tokens need to be deleted;
  + S(substitute): Substitued error , which means tokens need to be replaced;
  + W(word-order): Word-error error, which means tokens need to be re-ordered.
  
+ Linguistic tier (word granularity)：
  + We define 14 main types of linguistic errors (basically based on part-of-speech tags):
  
    ![error types](./pics/errors.PNG)


## Related works
+ We have used some technologies from this repo in the CTC-2021 evaluation task, and obtained the top-1 score. Please see: [CTC-report](https://github.com/HillZhang1999/CTC-Report).
+ Online demonstration platform of the baseline models: [GEC demo](http://139.224.234.18:5002/)。

## Contact
If you have any problems, feel free to contact me at hillzhang1999@qq.com.
