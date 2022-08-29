# MuCGEC Dataset Description
## Basic Information
The name of the dataset: MuCGEC（Multi-Reference Multi-Source Evaluation Dataset for Chinese Grammatical Error Correction）

This dataset contains：
+ dev set: 1137 sentences（12 unannotatable sentences）
+ test set: 6000 sentences（62 unannotatable sentences）

The dataset file: mucgec.zip. After you unzip the file, you should see:

+ MuCGEC_dev.txt: the dev set which contains the source sentences and the corresponding answers；
+ MuCGEC_test.txt: the test set which only contains the source sentences. The answers should be predicted by yourselfs;
+ example_pred_dev.txt: the example of the prediction results of the dev set；
+ filter_sentences.txt: the sentences that need to be filtered out from the train set (may overlap with test set)；
+ README.md: Instructions of the dataset, please read it carefully。

**The training data must filter out the sentences in filter_sentences.txt。**

## Data Format
### Dev Set

We provide a dev set (`MuCGEC_dev.txt` ) with answers for hyper-parameter tuning. The dev set is given in the format of TXT file, and the basic data format is: `id,\t, source sentence,\t, standard answer 1,\t, standard answer 2,...`, as follows:
```
1	因为在冰箱里没什么东西也做很好吃的菜。	即使在冰箱里没什么东西也能做很好吃的菜。	即使在冰箱里没什么东西，也能做很好吃的菜。
```

### Test Set
The test set shown below is also given in TXT format, and each column is separated by a `\t` symbol. However, the standard answers need to be predicted by the participants themselves. As follows:
```
2	纳税和帐单小一典。
```

Participants need to predict the unique modification result before submitting it and add it to the end of each line. As follows:
```
2	纳税和帐单小一典。  纳税和帐单小一点。
```

## Instructions
### Local Evaluation（Dev Set）

The official evaluation metric of MuCGEC is **the char-level F0.5 metric**.

The link of our evaluation tool: [https://github.com/HillZhang1999/MuCGEC/tree/main/scorers/ChERRANT](https://github.com/HillZhang1999/MuCGEC/tree/main/scorers/ChERRANT).

You can refer to the following link for the usage of our evaluation tool: [https://github.com/HillZhang1999/MuCGEC/tree/main/scorers/ChERRANT/README.md](https://github.com/HillZhang1999/MuCGEC/tree/main/scorers/ChERRANT/README.md).

The core steps:

1. Converting the standard answer file `MuCGEC_dev.txt` to M2 file `MuCGEC_dev.m2` through `parallel_to_m2.py` (it is only necessary for the first time and can be reused later);

```
python parallel_to_m2.py -f MuCGEC_dev.txt -o MuCGEC_dev.m2
```

2. Converting the prediction result file `example_pred_dev.txt` to M2 file `example_pred_dev.m2`  through `parallel_to_m2.py`；

```
python parallel_to_m2.py -f example_pred_dev.txt -o example_pred_dev.m2
```

3. Comparing `example_pred_dev.m2` and `MuCGEC_dev.m2` using `compare_m2_for_evaluation.py` to get the final evaluation results;
```
python compare_m2_for_evaluation.py -hyp example_pred_dev.m2 -ref MuCGEC_dev.m2
```

All prediction results need to be processed into the same format as `example_pred_dev.txt`.

If the evaluation process is correct, the word level indicator of the sample result file `example_pred_dev.txt` in the dev set should be:

```
=========== Span-Based Correction ============
TP      FP      FN      Prec    Rec     F0.5
1084    1635    3003    0.3987  0.2652  0.3622
==============================================
```

### Online Submission (Test Set)

Predict all sentences in the given test set file, and add the result to the end of each line. The specific format is `id, \t, source sentence, \t, predict result`. For example:

+ source file：
```
2	纳税和帐单小一典。
```

+ result file：
```
2	纳税和帐单小一典。	纳税和帐单小一点。
```

Still name the result file as  `MuCGEC_test.txt`, zip it, and upload it to the [Tianchi platform](https://tianchi.aliyun.com/dataset/dataDetail?dataId=131328)。

Note:
+ The zip file only needs to contain the prediction result txt file, and does not need to contain intermediate folders or other files.
+ The result file must be named as `mucgec_Test.txt`, and the content format must be the same as `example_pred_Dev.txt `.

## Detailed Metric Description

The indexes returned after online submission include:

+ TP: True Positive, which denotes the number of correct edits predicted by the model;
+ FP: False Positive, which denotes the number of wrong edits predicted by the model;
+ FN: False Negative, which denotes the number of correct edits that are not predicted by the model;
+ Precision: equal to TP/(TP+FP);
+ Recall: equal to TP/(TP+FN);
+ score: F0.5. Compared with the traditional F1 value, it pays more attention to the precision of the model and is a commonly used indicator in the GEC field.

Among the above indexes, the score (i.e., f0.5 value of the model) is the final ranking criterion.


## Citation

If you use any part of our work, please cite our paper:
```
@inproceedings{zhang-etal-2022-mucgec,
    title = "{MuCGEC}: a Multi-Reference Multi-Source Evaluation Dataset for Chinese Grammatical Error Correction",
    author = "Zhang, Yue and Li, Zhenghua and Bao, Zuyi and Li, Jiacheng and Zhang, Bo and Li, Chen and Huang, Fei and Zhang, Min",
    booktitle = "Proceedings of NAACL-HLT",
    year = "2022",
    address = "Online",
    publisher = "Association for Computational Linguistics"
```
