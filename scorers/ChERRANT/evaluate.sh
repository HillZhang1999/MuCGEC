RESULT_DIR=$1

INPUT_FILE=../../data/test_data/MuCGEC/MuCGEC-ALL/MuCGEC_ALL_Test.input
OUTPUT_FILE=$RESULT_DIR"/MuCGEC_test.output"
NLPCC18_OFFICIAL_M2=../../data/test_data/NLPCC18-orig/NLPCC18-orig.m2.pkunlp  # m2scorer, word-based
MuCGEC_ALL_M2=../../data/test_data/MuCGEC/MuCGEC-ALL/MuCGEC_ALL_Test.m2.char   # ChERRANT, char-based

if [ ! -f $RESULT_DIR"/MuCGEC_test.pred.m2.char" ]; then
  paste $INPUT_FILE $OUTPUT_FILE | awk '{print NR"\t"$p}' > $RESULT_DIR"/MuCGEC_test.pred.para"

  CUDA_VISIBLE_DEVICES=0 python parallel_to_m2.py -f $RESULT_DIR"/MuCGEC_test.pred.para" -o $RESULT_DIR"/MuCGEC_test.pred.m2.char" -g char -s all
fi

if [ ! -f $RESULT_DIR"/MuCGEC_test.output.pku_tok" ]; then
  head -n 2000 $OUTPUT_FILE | python ../../tools/segment/segment_pkunlp.py $RESULT_DIR"/MuCGEC_test.output.pku_tok" 
fi

echo "****************NLPCC18-official-MaxMatch****************"

python2 ../m2scorer/scripts/m2scorer.py $RESULT_DIR"/MuCGEC_test.output.pku_tok"  $NLPCC18_OFFICIAL_M2

echo "****************MuCGEC-NLPCC18-ChERRANT****************"

python compare_m2_for_evaluation.py -hyp $RESULT_DIR"/MuCGEC_test.pred.m2.char" -ref $MuCGEC_ALL_M2 --start 0 --end 2000

echo "****************MuCGEC-LANG8-ChERRANT****************"

python compare_m2_for_evaluation.py -hyp $RESULT_DIR"/MuCGEC_test.pred.m2.char" -ref $MuCGEC_ALL_M2 --start 2000 --end 4000

echo "****************MuCGEC-CGED-ChERRANT****************"

python compare_m2_for_evaluation.py -hyp $RESULT_DIR"/MuCGEC_test.pred.m2.char" -ref $MuCGEC_ALL_M2 --start 4000 --end 6000

echo "****************MuCGEC-ALL-ChERRANT****************"

python compare_m2_for_evaluation.py -hyp $RESULT_DIR"/MuCGEC_test.pred.m2.char" -ref $MuCGEC_ALL_M2
