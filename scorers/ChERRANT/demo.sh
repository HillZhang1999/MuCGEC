INPUT_FILE=./samples/demo.input
OUTPUT_FILE=./samples/demo.hyp
HYP_PARA_FILE=./samples/demo.hyp.para
HYP_M2_FILE=./samples/demo.hyp.m2.char
REF_M2_FILE=./samples/demo.ref.m2.char

# Step1. extract edits from hypothesis file.

paste $INPUT_FILE $OUTPUT_FILE | awk '{print NR"\t"$p}' > $HYP_PARA_FILE  # only for single hypothesis situation

python parallel_to_m2.py -f $HYP_PARA_FILE -o $HYP_M2_FILE -g char  # char-level evaluation

# Step2. compare hypothesis edits with reference edits.

python compare_m2_for_evaluation.py -hyp $HYP_M2_FILE -ref $REF_M2_FILE

# Note: you can also extract the reference edits yourself by using parallel_to_m2.py if you have reference sentences.
# You need to process the data into the following format: id \t source \t reference1 \t reference2 \t ... \n

# word-level evaluation
HYP_M2_FILE=./samples/demo.hyp.m2.word
REF_M2_FILE=./samples/demo.ref.m2.word
python parallel_to_m2.py -f $HYP_PARA_FILE -o $HYP_M2_FILE -g word  
python compare_m2_for_evaluation.py -hyp $HYP_M2_FILE -ref $REF_M2_FILE
