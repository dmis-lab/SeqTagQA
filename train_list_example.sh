export SEED=0
export NUM_ITER=`printf %02d $SEED`

#######
export METHOD=linear
export VERSION=8
export LM=biobert

# To train LMs on SQuAD, un-comment these lines
#export LANGMODEL_DIR=dmis-lab/biobert-v1.1
#export DATA_DIR=<Path to DATA>/DATA/SQuAD-SeqTagQA/squad-20201030
#export MAX_EPOCH=50
#export LEARN_RATE=5e-5
#export OUTPUT_DIR=<Output path>/20220130-squad_${METHOD}_lr${LEARN_RATE}_iter-${NUM_ITER}

# To train LMs on List-question dataset
export LANGMODEL_DIR=<Path to BioLM: it should be trained on SQuAD>
export DATA_DIR=<Path to DATA>/DATA/BioASQ-SeqTagQA/${VERSION}b-list-20201030
export MAX_EPOCH=400
export LEARN_RATE=5e-6
export OUTPUT_DIR=<Output path>/20220130-BioASQ-list${VERSION}b_${LM}-${METHOD}_lr${LEARN_RATE}_iter-${NUM_ITER}

echo $OUTPUT_DIR
mkdir $OUTPUT_DIR

python run_eqa.py --model_struct=${METHOD} \
 --do_train=true --do_eval=true --do_predict=true \
 --model_name_or_path=$LANGMODEL_DIR \
 --per_device_train_batch_size=18 --max_seq_length=512 \
 --train_file=${DATA_DIR}/train_dev.tsv --validation_file=${DATA_DIR}/test.tsv --test_file=${DATA_DIR}/test.tsv \
 --output_dir=$OUTPUT_DIR --save_total_limit=100 \
 --learning_rate=${LEARN_RATE} --seed=${SEED} \
 --num_train_epochs=${MAX_EPOCH} \
 --save_steps=4000

# To Detokenize the output
export CKPT=84000
export SUB_OUTPUT_DIR=${OUTPUT_DIR}/checkpoint-${CKPT}
python detokenize-list.py \
  --test_path=${DATA_DIR}/test.tsv \
  --predictions_path=${SUB_OUTPUT_DIR}/predictions.txt \
  --original_test_path=${DATA_DIR}/original_test.json \
  --output_dir=${SUB_OUTPUT_DIR}

# Example code for official BioASQ evaluation codes
#export EVAL_LIB=<Cloned official eval code path>/Evaluation-Measures
#java -Xmx10G -cp $CLASSPATH:$EVAL_LIB/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 \
#  <Path to DATA>/DATA/BioASQ-golden/${VERSION}B_total_golden.json ${SUB_OUTPUT_DIR}/NER_result_BioASQ.json >> $OUTPUT_DIR/total_BioASQ_eval.log
