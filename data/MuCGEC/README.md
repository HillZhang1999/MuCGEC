# MuCGEC数据集说明
## 基本信息
数据集名称：MuCGEC（Multi-Reference Multi-Source Evaluation Dataset for Chinese Grammatical Error Correction）

本次评测开放数据为：
+ 开发集：1137句（其中包含12句无法标注的句子）
+ 测试集：6000句（其中包含62句无法标注的句子）
数据集下载文件为：mucgec.zip，解压后包括：

+ MuCGEC_dev.txt：验证集，给出了每个句子的修改方式；
+ MuCGEC_test.txt：测试集，仅给出原句，但未给出修改方式，需要参赛人员自行预测；
+ example_pred_dev.txt：验证集预测结果示例；
+ filter_sentences.txt：需要参赛队员从训练集中筛除的句子；
+ README.md：数据集使用说明文件，请仔细阅读。

**训练数据必须剔除在filter_sentences.txt中的句子。**

## 数据格式
### 开发集

我们提供了给定答案的开发集（`MuCGEC_dev.txt`）以供模型调优。开发集以txt文件的格式给出，基本数据格式为：`id,\t,原句,\t,标准答案1,\t,标准答案2,...`，如下所示：
```
1	因为在冰箱里没什么东西也做很好吃的菜。	即使在冰箱里没什么东西也能做很好吃的菜。	即使在冰箱里没什么东西，也能做很好吃的菜。
```

### 测试集
测试集同样以txt格式给出，每列之间以`\t`制表符隔开，但未给定标准答案，需要自行预测。如下所示：
```
2	纳税和帐单小一典。
```

参赛队员提交前需要预测出唯一的修改结果，并添加到每一行的末尾。如下所示：
```
2	纳税和帐单小一典。  纳税和帐单小一点。
```

## 使用说明
### 本地评测（开发集）

本次评测的排名依据是**字级别的F0.5指标**，指标计算工具地址为：https://github.com/HillZhang1999/MuCGEC/tree/main/scorers/ChERRANT 。

评测工具使用方法可以参考：https://github.com/HillZhang1999/MuCGEC/tree/main/scorers/ChERRANT/README.md 。

核心步骤为：

1. 将标准答案文件`MuCGEC_dev.txt`通过`parallel_to_m2.py`脚本转换成为M2格式`MuCGEC_dev.m2`（仅首次评测时需要，后续复用即可）；
```
python parallel_to_m2.py -f MuCGEC_dev.txt -o MuCGEC_dev.m2
```
2. 将预测结果文件`example_pred_dev.txt`通过`parallel_to_m2.py`脚本转换成为M2格式`example_pred_dev.m2`；
```
python parallel_to_m2.py -f example_pred_dev.txt -o example_pred_dev.m2
```
3. 使用`compare_m2_for_evaluation.py`脚本对比`example_pred_dev.m2`和`MuCGEC_dev.m2`，得到最终的指标。
```
python compare_m2_for_evaluation.py -hyp example_pred_dev.m2 -ref MuCGEC_dev.m2
```

所有预测结果在进行评测前均需要处理成和`example_pred_dev.txt`一致的格式。

如果评测过程无误，样例结果文件`example_pred_dev.txt`在开发集的字级别指标应为：

```
=========== Span-Based Correction ============
TP      FP      FN      Prec    Rec     F0.5
1084    1635    3003    0.3987  0.2652  0.3622
==============================================
```

### 在线提交（测试集）

对给定的测试集文件中的所有句子进行预测，所得结果添加到每行末尾，具体格式为：`id,\t,原句,\t,预测结果`。如：

+ 原文件：
```
2	纳税和帐单小一典。
```

+ 结果文件：
```
2	纳税和帐单小一典。	纳税和帐单小一点。
```

将所得结果文件仍命名为`MuCGEC_test.txt`，打包为zip文件上传至[天池平台](https://tianchi.aliyun.com/dataset/dataDetail?dataId=131328)。

需要注意：
+ zip文件内仅需要包含唯一的预测结果txt文件，不需要包含中间文件夹或其他文件。
+ 结果文件必须命名成`MuCGEC_test.txt`，且内容格式与`example_pred_dev.txt`一致。

不附合上述要求的提交，将无法正确返回测试集指标。

## 指标说明

在线提交后返回的评测指标包括：

+ TP：True Positive，真正例，即模型预测的编辑中正确的数目；
+ FP：False Positive，伪正例，即模型预测的编辑中错误的数目；
+ FN：False Negative，伪负例，即模型未预测到的正确编辑的数目；
+ Precision：精确度，等于TP/(TP+FP)；
+ Recall：召回度，等于TP(TP+FN)；
+ score：F0.5值。与传统的F1值相比，更看重模型的精确度，是GEC领域常用的指标。

上述指标中，score（即模型的F0.5值）是所有榜单排名的最终依据。

## 引用

如您使用了我们的数据集或基线模型，请引用我们的论文：
```
@inproceedings{zhang-etal-2022-mucgec,
    title = "{MuCGEC}: a Multi-Reference Multi-Source Evaluation Dataset for Chinese Grammatical Error Correction",
    author = "Zhang, Yue and Li, Zhenghua and Bao, Zuyi and Li, Jiacheng and Zhang, Bo and Li, Chen and Huang, Fei and Zhang, Min",
    booktitle = "Proceedings of NAACL-HLT",
    year = "2022",
    address = "Online",
    publisher = "Association for Computational Linguistics"
```