# Seq2Edit的三个模型，更换了3个随机种子
RESULT_FILE1=$1
RESULT_FILE2=$2
RESULT_FILE3=$3

# Seq2Seq的三个模型，更换了3个随机种子
RESULT_FILE4=$4
RESULT_FILE5=$5
RESULT_FILE6=$6

# for ((i=1; i<=6; i++))
# do
#     THRESHOLD=$i
#     RESULT_DIR=./ensemble_results/3seq2edit_3seq2seq_threshold_$THRESHOLD
#     mkdir -p $RESULT_DIR

#     M2_OUTPUT_FILE=$RESULT_DIR/MuCGEC_test.m2_temp
#     OUTPUT_FILE=$RESULT_DIR/MuCGEC_test.output

#     python edit_ensemble.py --result_path $RESULT_FILE1 $RESULT_FILE2 $RESULT_FILE3 $RESULT_FILE4 $RESULT_FILE5 $RESULT_FILE6 --output_path $M2_OUTPUT_FILE --threshold $THRESHOLD

#     python m2convertor.py -f $M2_OUTPUT_FILE -o $OUTPUT_FILE
# done

THRESHOLD=4  # threshold 一般取 N/2+1，即当编辑出现次数大于等于该值时，才会保留
RESULT_DIR=./ensemble_results/3seq2edit_3seq2seq_threshold_$THRESHOLD
mkdir -p $RESULT_DIR

M2_OUTPUT_FILE=$RESULT_DIR/MuCGEC_test.m2_temp
OUTPUT_FILE=$RESULT_DIR/MuCGEC_test.output

python rule_ensemble.py --result_path $RESULT_FILE1 $RESULT_FILE2 $RESULT_FILE3 $RESULT_FILE4 $RESULT_FILE5 $RESULT_FILE6 --output_path $M2_OUTPUT_FILE --threshold $THRESHOLD

python m2convertor.py -f $M2_OUTPUT_FILE -o $OUTPUT_FILE
