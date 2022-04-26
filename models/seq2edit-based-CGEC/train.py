# -*- coding: utf-8
import argparse
import logging
import os
import time
from random import seed
import torch
from allennlp.data import allennlp_collate
from allennlp.data.dataloader import PyTorchDataLoader
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.modules import Embedding
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
from allennlp.training import GradientDescentTrainer
from allennlp.training.learning_rate_schedulers import ReduceOnPlateauLearningRateScheduler
from torch.utils.data import DataLoader
from allennlp.training.optimizers import AdamOptimizer
from gector.datareader import Seq2LabelsDatasetReader
from gector.seq2labels_model import Seq2Labels
from allennlp.training.tensorboard_writer import TensorboardWriter

def fix_seed(s):
    """
    固定随机种子
    """
    torch.manual_seed(s)
    seed(s)

def get_token_indexers(model_name):
    """
    获取token编号器（主要是不同预训练BERT模型的子词算法不同，因而index策略也不同）
    :param model_name: 模型名称
    :return: 返回token编号器
    """
    bert_token_indexer = PretrainedTransformerIndexer(model_name=model_name, namespace="bert")
    return {'bert': bert_token_indexer}

def get_token_embedders(model_name, tune_bert=False):
    """
    获取token嵌入器
    :param model_name: 模型名称
    :param tune_bert: 是否微调
    :return: token文本域嵌入器
    """
    take_grads = True if tune_bert > 0 else False
    bert_token_emb = PretrainedTransformerEmbedder(model_name=model_name, last_layer_only=True,
                                                   train_parameters=take_grads)
    token_embedders = {'bert': bert_token_emb}

    text_filed_emd = BasicTextFieldEmbedder(token_embedders=token_embedders)
    return text_filed_emd

def build_data_loaders(
        data_set: AllennlpDataset,
        batch_size: int,
        num_workers: int,
        shuffle: bool,
        batches_per_epoch = None
):
    """
    创建数据载入器
    :param batches_per_epoch:
    :param data_set: 数据集对象
    :param batch_size: batch大小
    :param num_workers: 同时使用多少个线程载入数据
    :param shuffle: 是否打乱训练集
    :return: 训练集、开发集、测试集数据载入器
    """
    return PyTorchDataLoader(data_set, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                      collate_fn=allennlp_collate, batches_per_epoch=batches_per_epoch)

def get_data_reader(model_name, max_len, skip_correct=False, skip_complex=0,
                    test_mode=False, tag_strategy="keep_one",
                    broken_dot_strategy="keep",
                    tn_prob=0, tp_prob=1, ):
    token_indexers = get_token_indexers(model_name)
    reader = Seq2LabelsDatasetReader(token_indexers=token_indexers,
                                     max_len=max_len,
                                     skip_correct=skip_correct,
                                     skip_complex=skip_complex,
                                     test_mode=test_mode,
                                     tag_strategy=tag_strategy,
                                     broken_dot_strategy=broken_dot_strategy,
                                     lazy=True,
                                     tn_prob=tn_prob,
                                     tp_prob=tp_prob)
    return reader


def get_model(model_name, vocab, tune_bert=False, predictor_dropout=0,
              label_smoothing=0.0,
              confidence=0,
              model_dir="",
              log=None):
    token_embs = get_token_embedders(model_name, tune_bert=tune_bert)
    model = Seq2Labels(vocab=vocab,
                       text_field_embedder=token_embs,
                       predictor_dropout=predictor_dropout,
                       label_smoothing=label_smoothing,
                       confidence=confidence,
                       model_dir=model_dir,
                       cuda_device=args.cuda_device,
                       dev_file=args.dev_set,
                       logger=log,
                       vocab_path=args.vocab_path,
                       weight_name=args.weights_name,
                       save_metric=args.save_metric
                       )
    return model


def main(args):
    fix_seed(args.seed)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    logger = logging.getLogger(__file__)
    logger.setLevel(level=logging.INFO)
    start_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    handler = logging.FileHandler(args.model_dir + '/logs_{:s}.txt'.format(str(start_time)))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    weights_name = args.weights_name
    reader = get_data_reader(weights_name, args.max_len, skip_correct=bool(args.skip_correct),
                             skip_complex=args.skip_complex,
                             test_mode=False,
                             tag_strategy=args.tag_strategy,
                             tn_prob=args.tn_prob,
                             tp_prob=args.tp_prob)
    train_data = reader.read(args.train_set)
    dev_data = reader.read(args.dev_set)

    default_tokens = [DEFAULT_OOV_TOKEN, DEFAULT_PADDING_TOKEN]
    namespaces = ['labels', 'd_tags']
    tokens_to_add = {x: default_tokens for x in namespaces}

    # build vocab
    if args.vocab_path:
        vocab = Vocabulary.from_files(args.vocab_path)
    else:
        vocab = Vocabulary.from_instances(train_data,
                                          min_count={"labels": 5},
                                          tokens_to_add=tokens_to_add)
        vocab.save_to_files(args.vocab_path)

    print("Data is loaded")
    logger.info("Data is loaded")

    model = get_model(weights_name, vocab,
                      tune_bert=args.tune_bert,
                      predictor_dropout=args.predictor_dropout,
                      label_smoothing=args.label_smoothing,
                      model_dir=os.path.join(args.model_dir, args.model_name + '.th'),
                      log=logger)

    device = torch.device("cuda:" + str(args.cuda_device) if int(args.cuda_device) >= 0 else "cpu")
    if args.pretrain:  # 只加载部分预训练模型
        pretrained_dict = torch.load(os.path.join(args.pretrain_folder, args.pretrain + '.th'), map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print('load pretrained model')
        logger.info('load pretrained model')

    model = model.to(device)
    print("Model is set")
    logger.info("Model is set")

    parameters = [
        (n, p)
        for n, p in model.named_parameters() if p.requires_grad
    ]
    
    # 使用Adam算法进行SGD
    optimizer = AdamOptimizer(parameters, lr=args.lr, betas=(0.9, 0.999))
    scheduler = ReduceOnPlateauLearningRateScheduler(optimizer)
    train_data.index_with(vocab)
    dev_data.index_with(vocab)
    tensorboardWriter = TensorboardWriter(args.model_dir)
    trainer = GradientDescentTrainer(
        model=model,
        data_loader=build_data_loaders(train_data, batch_size=args.batch_size, num_workers=0, shuffle=False, batches_per_epoch=args.updates_per_epoch),
        validation_data_loader=build_data_loaders(dev_data, batch_size=args.batch_size, num_workers=0, shuffle=False),
        num_epochs=args.n_epoch,
        optimizer=optimizer,
        patience=args.patience,
        validation_metric=args.save_metric,
        cuda_device=device,
        num_gradient_accumulation_steps=args.accumulation_size,
        learning_rate_scheduler=scheduler,
        tensorboard_writer=tensorboardWriter,
        use_amp=True  # 混合精度训练，如果显卡不支持请设为false
    )
    print("Start training")
    print('\nepoch: 0')
    logger.info("Start training")
    logger.info('epoch: 0')
    trainer.train()


if __name__ == '__main__':
    # read parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set',
                        help='Path to the train data',
                        required=True)  # 训练集路径（带标签格式）
    parser.add_argument('--dev_set',
                        help='Path to the dev data',
                        required=True)  # 开发集路径（带标签格式）
    parser.add_argument('--model_dir',
                        help='Path to the model dir',
                        required=True)  # 模型保存路径
    parser.add_argument('--model_name',
                        help='The name of saved checkpoint',
                        required=True)  # 模型名称
    parser.add_argument('--vocab_path',
                        help='Path to the model vocabulary directory.'
                             'If not set then build vocab from data',
                        default="./data/output_vocabulary_chinese_char_hsk+lang8_5")  # 词表路径
    parser.add_argument('--batch_size',
                        type=int,
                        help='The size of the batch.',
                        default=256)  # batch大小（句子数目）
    parser.add_argument('--max_len',
                        type=int,
                        help='The max sentence length'
                             '(all longer will be truncated)',
                        default=200)  # 最大输入长度，过长句子将被截断
    parser.add_argument('--target_vocab_size',
                        type=int,
                        help='The size of target vocabularies.',
                        default=1000)  # 词表规模（生成词表时才需要）
    parser.add_argument('--n_epoch',
                        type=int,
                        help='The number of epoch for training model.',
                        default=2)  # 训练轮数
    parser.add_argument('--patience',
                        type=int,
                        help='The number of epoch with any improvements'
                             ' on validation set.',
                        default=3)  # 早停轮数
    parser.add_argument('--skip_correct',
                        type=int,
                        help='If set than correct sentences will be skipped '
                             'by data reader.',
                        default=1)  # 是否跳过正确句子
    parser.add_argument('--skip_complex', 
                        type=int,
                        help='If set than complex corrections will be skipped '
                             'by data reader.',
                        choices=[0, 1, 2, 3, 4, 5],
                        default=0)  # 是否跳过复杂句子
    parser.add_argument('--tune_bert',
                        type=int,
                        help='If more then 0 then fine tune bert.',
                        default=0)  # 是否微调bert
    parser.add_argument('--tag_strategy',
                        choices=['keep_one', 'merge_all'],
                        help='The type of the data reader behaviour.',
                        default='keep_one')  # 标签抽取策略，前者每个位置只保留一个标签，后者保留所有标签
    parser.add_argument('--lr',
                        type=float,
                        help='Set initial learning rate.',
                        default=1e-3)  # 初始学习率
    parser.add_argument('--predictor_dropout',
                        type=float,
                        help='The value of dropout for predictor.',
                        default=0.0)  # dropout率（除bert以外部分）
    parser.add_argument('--label_smoothing',
                        type=float,
                        help='The value of parameter alpha for label smoothing.',
                        default=0.0)  # 标签平滑
    parser.add_argument('--tn_prob',
                        type=float,
                        help='The probability to take TN from data.',
                        default=0)  # 保留正确句子的比例
    parser.add_argument('--tp_prob', 
                        type=float,
                        help='The probability to take TP from data.',
                        default=1)  # 保留错误句子的比例
    parser.add_argument('--pretrain_folder',
                        help='The name of the pretrain folder.',
                        default=None)  # 之前已经训练好的checkpoint的文件夹
    parser.add_argument('--pretrain',  
                        help='The name of the pretrain weights in pretrain_folder param.',
                        default=None)  # 之前已经训练好的checkpoint名称
    parser.add_argument('--cuda_device',
                        help='The number of GPU',
                        default=0)  # 使用GPU编号
    parser.add_argument('--accumulation_size',
                        type=int,
                        help='How many batches do you want accumulate.',
                        default=1)  # 梯度累积
    parser.add_argument('--weights_name',
                        type=str,
                        default="chinese-struct-bert")  # 预训练语言模型路径
    parser.add_argument('--save_metric',
                        type=str,
                        choices=["+labels_accuracy", "+labels_accuracy_except_keep"],
                        default="+labels_accuracy")  # 模型保存指标
    parser.add_argument('--updates_per_epoch',
                        type=int,
                        default=None)  # 每个epoch更新次数
    parser.add_argument('--seed',
                        type=int,
                        default=1)  # 随机种子
    args = parser.parse_args()
    main(args)
