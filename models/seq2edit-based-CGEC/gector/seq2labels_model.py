"""Basic model. Predicts tags for every token
GECToR真正意义上的模型，对一个源句子序列使用Transformer做encode，然后在每个token处使用MLP预测最可能的编辑label
"""
# -*- coding: utf-8

import os
from typing import *

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed, TextFieldEmbedder, ConditionalRandomField
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, tiny_value_of_dtype
from gector.seq2labels_metric import Seq2LabelsMetric
from overrides import overrides
from torch.nn.modules.linear import Linear
from allennlp.nn import util


@Model.register("seq2labels")
class Seq2Labels(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 predictor_dropout=0.0,
                 labels_namespace: str = "labels",
                 detect_namespace: str = "d_tags",
                 label_smoothing: float = 0.0,
                 confidence: float = 0.0,
                 model_dir: str = "",
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 hidden_layers: int = 0,
                 hidden_dim: int = 512,
                 cuda_device: int = 0,
                 dev_file: str = None,
                 logger=None,
                 vocab_path: str = None,
                 weight_name: str = None,
                 save_metric: str = "dev_m2",
                 beta: float = None,
                 ) -> None:
        """
        seq2labels模型的构造函数
        :param vocab: 词典对象
        :param text_field_embedder: 嵌入器，这里采用预训练的类bert模型作为嵌入器对token进行embedding（即论文模型结构中的Transformer-Encoder）
        :param predictor_dropout: 预测器dropout概率（防止过拟合）
        :param labels_namespace: labels的命名空间（GECToR解码端的labels输出指的是编辑label，如$KEEP等）
        :param detect_namespace: detect的命令空间（GECToR解码端的d_tags输出指的是探测当前token是否出错的一个二分类标签）
        :param label_smoothing: 一个正则化的trick，减少分类错误带来的惩罚
        :param confidence:  $KEEP标签的偏差项
        :param model_dir:  模型保存路径
        :param initializer: 初始化器
        :param regularizer: 正则化器
        """
        super(Seq2Labels, self).__init__(vocab, regularizer)
        self.save_metric = save_metric
        self.weight_name = weight_name
        self.cuda_device = cuda_device
        self.device = torch.device("cuda:" + str(cuda_device) if int(cuda_device) >= 0 else "cpu")
        self.label_namespaces = [labels_namespace,
                                 detect_namespace]  # 需要解码预测的标签的命名空间
        self.text_field_embedder = text_field_embedder
        self.num_labels_classes = self.vocab.get_vocab_size(labels_namespace)
        self.num_detect_classes = self.vocab.get_vocab_size(detect_namespace)
        self.label_smoothing = label_smoothing
        self.confidence = confidence
        self.incorr_index = self.vocab.get_token_index("INCORRECT",
                                                       namespace=detect_namespace)  # 获取INCORRECT标签的索引
        self.vocab_path = vocab_path
        self.best_metric = 0.0
        self.epoch = 0
        self.model_dir = model_dir
        self.logger = logger
        self.beta = beta
        self.predictor_dropout = TimeDistributed(torch.nn.Dropout(predictor_dropout))
        # TimeDistributed模块是临时将(batch_size, time_steps, [rest])的time_steps维度暂时降维，变为(batch_size * time_steps, [rest])的矩阵，提供给某个模块（如维度为([rest], output_dim)的Linear层），
        # 然后再将该模块的输出(batch_size * time_steps, output_dim)变为(batch_size, time_steps, output_dim)的形式
        # TimeDistributed层在需要并行解码时非常常用。因为decoder端的MLP等需要同时应用在当前序列各time_step上，同步解码。
        self.dev_file = dev_file
        self.tag_labels_hidden_layers = []
        self.tag_detect_hidden_layers = []
        input_dim = text_field_embedder.get_output_dim()
        if hidden_layers > 0:
            self.tag_labels_hidden_layers.append(TimeDistributed(
                Linear(input_dim,
                       hidden_dim)).cuda(self.device))
            self.tag_detect_hidden_layers.append(TimeDistributed(
                Linear(input_dim,
                       hidden_dim)).cuda(self.device))
            for _ in range(hidden_layers - 1):
                self.tag_labels_hidden_layers.append(TimeDistributed(
                    Linear(hidden_dim, hidden_dim)).cuda(self.device))
                self.tag_detect_hidden_layers.append(TimeDistributed(
                    Linear(hidden_dim, hidden_dim)).cuda(self.device))
            self.tag_labels_projection_layer = TimeDistributed(
                Linear(hidden_dim,
                       self.num_labels_classes)).cuda(self.device)  # 编辑label预测线性投影层
            self.tag_detect_projection_layer = TimeDistributed(
                Linear(hidden_dim,
                       self.num_detect_classes)).cuda(self.device)  # 编辑label预测线性投影层
        else:
            self.tag_labels_projection_layer = TimeDistributed(
                Linear(input_dim, self.num_labels_classes)).to(self.device)  # 编辑label预测线性投影层
            self.tag_detect_projection_layer = TimeDistributed(
                Linear(input_dim, self.num_detect_classes)).to(self.device)  # 是否错误tag预测线性投影层

        # self.metrics = {"accuracy": CategoricalAccuracy()}
        self.metric = Seq2LabelsMetric()
        # 模型的预测评价指标采用分类准确率
        self.UNKID = self.vocab.get_vocab_size("labels") - 2

        initializer(self)

    @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None,
                d_tags: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        encoded_text = self.text_field_embedder(tokens)  # 返回bert模型的输出。维度：[batch_size,seq_len,encoder_output_dim]
        batch_size, sequence_length, _ = encoded_text.size()
        mask = get_text_field_mask(tokens)  # [batch_size,seq_len]  # 返回mask标记（防止因句子长度不一致而padding的影响）

        # 训练模式（训练集）
        if self.training:
            ret_train = self.decode(encoded_text, batch_size, sequence_length, mask, labels, d_tags, metadata)
            _loss = ret_train['loss']
            output_dict = {'loss': _loss}
            logits_labels = ret_train["logits_labels"]
            logits_d = ret_train["logits_d_tags"]
            self.metric(logits_labels, labels, logits_d, d_tags, mask.float())
            return output_dict

        # 评测模式（开发集）
        training_mode = self.training
        self.eval()
        with torch.no_grad():
            ret_train = self.decode(encoded_text, batch_size, sequence_length, mask, labels, d_tags, metadata)
        self.train(training_mode)
        logits_labels = ret_train["logits_labels"]
        logits_d = ret_train["logits_d_tags"]
        if labels is not None and d_tags is not None:  # 如果没有提供golden labels标签和d_tags标签，那么就是预测模式（测试集），无需计算accuracy
            self.metric(logits_labels, labels, logits_d, d_tags, mask.float())
        return ret_train

    def decode(self, encoded_text: torch.LongTensor = None,
               batch_size: int = 0,
               sequence_length: int = 0,
               mask: torch.LongTensor = None,
               labels: torch.LongTensor = None,
               d_tags: torch.LongTensor = None,
               metadata: List[Dict[str, Any]] = None) -> Dict:
        if self.tag_labels_hidden_layers:
            encoded_text_labels = encoded_text.clone().to(self.device)
            for layer in self.tag_labels_hidden_layers:
                encoded_text_labels = layer(encoded_text_labels)
            logits_labels = self.tag_labels_projection_layer(
                self.predictor_dropout(
                    encoded_text_labels))  # 用一个简单的全连接层预测当前token处的label得分，[batch_size,seq_len,num_labels_classes]
            for layer in self.tag_detect_hidden_layers:
                encoded_text = layer(encoded_text)
            logits_d = self.tag_detect_projection_layer(
                self.predictor_dropout(
                    encoded_text))  # 用一个简单的全连接层预测当前token处的label得分，[batch_size,seq_len,num_labels_classes]
        else:
            logits_labels = self.tag_labels_projection_layer(
                self.predictor_dropout(
                    encoded_text))  # 用一个简单的全连接层预测当前token处的label得分，[batch_size,seq_len,num_labels_classes]
            logits_d = self.tag_detect_projection_layer(
                encoded_text)  # 用一个简单的全连接层预测当前token处的label得分，[batch_size,seq_len,num_detect_classes]

        class_probabilities_labels = F.softmax(logits_labels, dim=-1).view(
            [batch_size, sequence_length, self.num_labels_classes])  # 利用Softmax函数，将得分转为概率

        class_probabilities_d = F.softmax(logits_d, dim=-1).view(
            [batch_size, sequence_length, self.num_detect_classes])  # 利用Softmax函数，将得分转为概率

        error_probs = class_probabilities_d[:, :,
                      self.incorr_index] * mask  # [batch_size,sen_qen]点乘[batch_size,sen_qen]=[batch_size,sen_qen]，获得每个句子的每个token错误的概率
        incorr_prob = torch.max(error_probs, dim=-1)[
            0]  # [batch_size]:取每个句子所有token的错误概率最大者，作为此句子的错误概率（用于min_error_probability的trick）

        if self.confidence > 0:  # 给$KEEP标签添加一个置信度bias，优先预测$KEEP，防止模型过多地纠错，属于一个小trick
            probability_change = [self.confidence] + [0] * (self.num_labels_classes - 1)
            offset = torch.FloatTensor(probability_change).repeat(
                (batch_size, sequence_length, 1)).to(self.device)
            class_probabilities_labels += util.move_to_device(offset, self.device)

        # 输出前向传播计算的结果
        output_dict = {"logits_labels": logits_labels,
                       "logits_d_tags": logits_d,
                       "class_probabilities_labels": class_probabilities_labels,
                       "class_probabilities_d_tags": class_probabilities_d,
                       "max_error_probability": incorr_prob}
        # 以上是训练阶段和预测阶段共享的过程

        # 下面时训练阶段独占的过程，因为需要计算loss进行反向传播更新参数
        if labels is not None and d_tags is not None:  # 如果没有提供golden labels标签和d_tags标签，那么就是预测模式，无需计算loss
            loss_labels = sequence_cross_entropy_with_logits(logits_labels, labels, mask,
                                                             label_smoothing=self.label_smoothing)
            loss_d = sequence_cross_entropy_with_logits(logits_d, d_tags, mask)
            output_dict["loss"] = loss_labels + loss_d

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        获取模型的评级指标
        :param reset: 是否重置模型的评价指标
        :return: 模型的评级指标
        """
        metrics_to_return = self.metric.get_metric(reset)
        # reset设为True，则会将模型当前累计的评价指标数据清空。一般来说，allennlp的训练器会在每个epoch结束时的那个batch，调用reset为True的get_metrics，以便在下一轮重新计算指标。
        if self.metric is not None and not self.training:
            if reset:
                labels_accuracy = float(metrics_to_return['labels_accuracy'].item())
                print('The accuracy of predicting for edit labels is: ' + str(labels_accuracy))
                labels_accuracy_except_keep = float(metrics_to_return['labels_accuracy_except_keep'].item())
                self.logger.info('The accuracy of predicting for edit labels is: ' + str(labels_accuracy))
                print('The accuracy of predicting for edit labels except keep label is: ' + str(
                    labels_accuracy_except_keep))
                self.logger.info('The accuracy of predicting for edit labels except keep label is: ' + str(
                    labels_accuracy_except_keep))
                tmp_model_dir = "/".join(self.model_dir.split('/')[:-1]) + "/Temp_Model.th" 
                self.save(tmp_model_dir)
                if self.save_metric == "+labels_accuracy":
                    if self.best_metric <= labels_accuracy:
                        print('(best)Saving Model...')
                        self.logger.info('(best)Saving Model...')
                        self.best_metric = labels_accuracy
                        self.save(self.model_dir)
                    print('best labels_accuracy till now:' + str(self.best_metric))
                    self.logger.info('best labels_accuracy till now:' + str(self.best_metric))
                elif self.save_metric == "+labels_accuracy_except_keep":
                    if self.best_metric <= labels_accuracy_except_keep:
                        print('(best)Saving Model...')
                        self.logger.info('(best)Saving Model...')
                        self.best_metric = labels_accuracy_except_keep
                        self.save(self.model_dir)
                    print('best labels_accuracy_except_keep till now:' + str(self.best_metric))
                    self.logger.info('best labels_accuracy_except_keep till now:' + str(self.best_metric))
                else:
                    raise NotImplementedError("Wrong metric!")
                self.epoch += 1
                print(f'\nepoch: {self.epoch}')
                self.logger.info(f'epoch: {self.epoch}')

        return metrics_to_return

    def save(self, model_dir):
        """
        保存模型
        :param model_dir: 模型保存文件夹
        """
        with open(model_dir, 'wb') as f:
            torch.save(self.state_dict(), f)
        print("Model is dumped")
        self.logger.info("Model is dumped")
