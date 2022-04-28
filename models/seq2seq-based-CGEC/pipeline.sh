# Step 1. Data preprocessing
DATA_DIR=./exp_data/lang8+hsk_only_erroneous
mkdir -p $DATA_DIR

TRAIN_SRC_FILE=../../data/train_data/lang8+hsk/train.src_only_erroneous  # 每行一个病句
TRAIN_TGT_FILE=../../data/train_data/lang8+hsk/train.tgt_only_erroneous  # 每行一个正确句子，和病句一一对应
if [ ! -f $DATA_DIR"/train.json" ]; then
    python ./utils.py $TRAIN_SRC_FILE $TRAIN_TGT_FILE $DATA_DIR"/train.json"
fi

VALID_SRC_FILE=../../data/valid_data/MuCGEC_CGED_Dev.src  # 每行一个病句
VALID_TGT_FILE=../../data/valid_data/MuCGEC_CGED_Dev.tgt  # 每行一个正确句子，和病句一一对应
if [ ! -f $DATA_DIR"/valid.json" ]; then
    python ./utils.py $VALID_SRC_FILE $VALID_TGT_FILE $DATA_DIR"/valid.json"
fi

# Step 2. Training
SEED=2021
PRETRAIN_MODEL=fnlp/bart-large-chinese
MODEL_DIR=./exps/lang8+lang8_bart-large-chinese
TASK_NAME=gec
CUDA_DEVICE=0,1,2,3,4,5,6,7

mkdir -p $MODEL_DIR/$TASK_NAME/run_$SEED/src_bak
cp ./pipeline.sh $MODEL_DIR/$TASK_NAME/run_$SEED/src_bak
cp train.py $MODEL_DIR/$TASK_NAME/run_$SEED/src_bak
cp predict.py $MODEL_DIR/$TASK_NAME/run_$SEED/src_bak

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u train.py \
    --do_train \
    --do_eval \
    --model_path $PRETRAIN_MODEL \
    --save_path $MODEL_DIR \
    --task $TASK_NAME \
    --data_dir $DATA_DIR \
    --seed $SEED \

# Step 3. Inference
MODEL_PATH=./exps/seq2seq_lang8
RESULT_DIR=$MODEL_PATH/results
mkdir -p $RESULT_DIR
INPUT_FILE=../../data/test_data/MuCGEC/MuCGEC-ALL/MuCGEC_ALL_Test.input # 输入文件（无需分字）
OUTPUT_FILE=$RESULT_DIR"/MuCGEC_test.output" # 输出文件

echo "Generating..."
SECONDS=0
CUDA_VISIBLE_DEVICES=0 python -u predict.py \
    --model_path $MODEL_PATH \
    --input_path $INPUT_FILE \
    --output_path $OUTPUT_FILE ;

echo "Generating Finish!"
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

