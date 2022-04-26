import torch
from allennlp.training.metrics.metric import Metric
from overrides import overrides

@Metric.register("Seq2LabelsMetric")
class Seq2LabelsMetric(Metric):
    """
    计算评价指标
    """

    def __init__(self, eps=1e-8):
        super().__init__()

        self.eps = eps
        self.count_token = 0
        self.labels_without_keep = 0
        self.labels_correct = 0
        self.d_tags_correct = 0
        self.labels_without_keep_correct = 0

    def __repr__(self):
        s = f"labels_accuracy: {self.Labels_Accuracy}, d_tags_accuracy: {self.Tags_Accuracy}, labels_accuracy_except_keep: {self.Labels_Accuracy_Except_Keep}.\n"
        return s

    @overrides
    def __call__(self, logits_labels, labels, logits_d, d_tags, mask=None, crf_result=None):
        logits_labels, labels, logits_d, d_tags, mask = self.detach_tensors(logits_labels, labels, logits_d, d_tags,
                                                                            mask)
        num_labels = logits_labels.size(-1)
        num_tags = logits_d.size(-1)

        logits_labels = logits_labels.view((-1, num_labels))
        labels = labels.view(-1).long()
        if crf_result is not None:
            argmax_labels = crf_result.view(-1).unsqueeze(-1)
        else:
            argmax_labels = logits_labels.max(-1)[1].unsqueeze(-1)
        # print(argmax_labels.shape, labels.shape)
        correct_labels = argmax_labels.eq(labels.unsqueeze(-1)).float()
        labels_ueq_keep = labels.unsqueeze(-1) != 0
        correct_labels_ueq_keep = (argmax_labels.eq(labels.unsqueeze(-1)) & labels_ueq_keep).float()
        labels_ueq_keep = labels_ueq_keep.float()

        logits_d = logits_d.view((-1, num_tags))
        d_tags = d_tags.view(-1).long()
        argmax_tags = logits_d.max(-1)[1].unsqueeze(-1)
        correct_tags = argmax_tags.eq(d_tags.unsqueeze(-1)).float()

        if mask is not None:
            correct_labels *= mask.view(-1, 1)
            correct_tags *= mask.view(-1, 1)
            correct_labels_ueq_keep *= mask.view(-1, 1)
            _total_count = mask.sum()
            labels_ueq_keep *= mask.view(-1, 1)
            _labels_without_keep = labels_ueq_keep.sum()
        else:
            _total_count = torch.tensor(labels.numel())
            _labels_without_keep = labels_ueq_keep.sum()
        _correct_labels_count = correct_labels.sum()
        _correct_tags_count = correct_tags.sum()
        _correct_labels_ueq_keep_count = correct_labels_ueq_keep.sum()
        self.count_token += _total_count
        self.labels_without_keep += _labels_without_keep
        self.labels_correct += _correct_labels_count
        self.d_tags_correct += _correct_tags_count
        self.labels_without_keep_correct += _correct_labels_ueq_keep_count

    @overrides
    def reset(self):
        self.count_token = 0
        self.labels_without_keep = 0
        self.labels_correct = 0
        self.d_tags_correct = 0
        self.labels_without_keep_correct = 0

    @property
    def Labels_Accuracy(self):
        """
        编辑标签预测准确率
        """
        return self.labels_correct / (self.count_token + self.eps)

    @property
    def Tags_Accuracy(self):
        """
        token是否有误标签预测准确率
        """
        return self.d_tags_correct / (self.count_token + self.eps)

    @property
    def Labels_Accuracy_Except_Keep(self):
        """
        编辑标签预测准确率（除$Keep标签外）
        """
        return self.labels_without_keep_correct / (self.labels_without_keep + self.eps)

    @property
    def Total_Accuracy(self):
        """
        编辑标签预测准确率
        """
        return self.Labels_Accuracy + self.Tags_Accuracy + self.Labels_Accuracy_Except_Keep

    def get_metric(self, reset: bool = False, model_object=None):
        ret = {"labels_accuracy": self.Labels_Accuracy, "d_tags_accuracy": self.Tags_Accuracy, "labels_accuracy_except_keep": self.Labels_Accuracy_Except_Keep, "total_accuracy": self.Total_Accuracy}
        if reset:
            self.reset()
        return ret
