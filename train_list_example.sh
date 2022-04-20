export SEED=0
export NUM_ITER=`printf %02d $SEED`

#######
export METHOD=linear
export VERSION=8
export LM=biobert

# SQuAD
#export BIOBERT_DIR=dmis-lab/biobert-v1.1
#export NER_DIR=<Path to DATA>/DATA/SQuAD-SeqTagQA/squad-20201030
#export MAX_EPOCH=50
#export LEARN_RATE=5e-5
#export OUTPUT_DIR=<Output path>/20220130-squad_${METHOD}_lr${LEARN_RATE}_iter-${NUM_ITER}

# List
export BIOBERT_DIR=<Path to BioLM: it should be trained on SQuAD>
export NER_DIR=<Path to DATA>/DATA/BioASQ-SeqTagQA/${VERSION}b-list-20201030
export MAX_EPOCH=400
export LEARN_RATE=5e-6
export OUTPUT_DIR=<Output path>/20220130-BioASQ-list${VERSION}b_${LM}-${METHOD}_lr${LEARN_RATE}_iter-${NUM_ITER}

echo $OUTPUT_DIR
mkdir $OUTPUT_DIR

python run_eqa.py --model_struct=${METHOD} \
 --do_train=true --do_eval=true --do_predict=true \
 --model_name_or_path=$BIOBERT_DIR \
 --per_device_train_batch_size=18 --max_seq_length=512 \
 --train_file=${NER_DIR}/train_dev.tsv --validation_file=${NER_DIR}/test.tsv --test_file=${NER_DIR}/test.tsv \
 --output_dir=$OUTPUT_DIR --save_total_limit=100 \
 --learning_rate=${LEARN_RATE} --seed=${SEED} \
 --num_train_epochs=${MAX_EPOCH} \
 --save_steps=4000
 
