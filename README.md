## Sequence Tagging for Biomedical Extractive Question Answering

Source codes and resources for "Sequence Tagging for Biomedical Extractive Question Answering."

Status: `Submitted to a journal. Waiting for pier-reviews.`

Pre-print (2021): https://arxiv.org/abs/2104.07535 (To be updated after the journal review)

------




## How to train/evaluate

For datasets (including biomedical questions), please download them from: [here](https://drive.google.com/file/d/1m0GnVlKqvUHfDdpZ9KDDor5EiIqhPwp3/view?usp=sharing)

For example training and evaluation bash script: please see [`train_list_example.sh`](./train_list_example.sh)

### Detailed instructions
#### Training
Following instructions are an example scripts of training LM on BioASQ8b dataset.

At the first part, we set environmental variables.
(We assume that you downloaded and decompressed the datasets in `$HOME/DATA` )
```bash
export SEED=0
export VERSION=8

# To train LMs on List-question dataset
export LANGMODEL_DIR=<Path to BioLM: it should be trained on SQuAD>
export DATA_DIR=$HOME/DATA/BioASQ-SeqTagQA/${VERSION}b-list-20201030
export MAX_EPOCH=400
export LEARN_RATE=5e-6
export OUTPUT_DIR=bioasq8b-list
mkdir $OUTPUT_DIR
```

The following lines will run the training of the model.:
```bash
python run_eqa.py --model_struct=linear \
 --do_train=true --do_eval=true --do_predict=false \
 --model_name_or_path=$LANGMODEL_DIR \
 --per_device_train_batch_size=18 --max_seq_length=512 \
 --train_file=${DATA_DIR}/train_dev.tsv --validation_file=${DATA_DIR}/test.tsv --test_file=${DATA_DIR}/test.tsv \
 --output_dir=$OUTPUT_DIR --save_total_limit=100 \
 --learning_rate=${LEARN_RATE} --seed=${SEED} \
 --num_train_epochs=${MAX_EPOCH} \
 --save_steps=4000
```

#### Testing
Then select the latest checkpoint and predict test dataset using the checkpoint.
```bash
export CKPT=84000
export SUB_OUTPUT_DIR=${OUTPUT_DIR}/checkpoint-${CKPT}

python run_eqa.py --model_struct=${METHOD} \
  --do_train=false --do_eval=false --do_predict=true \
  --model_name_or_path=$SUB_OUTPUT_DIR \
  --per_device_train_batch_size=18 --per_device_eval_batch_size=32 --max_seq_length=512 \
  --test_file=${DATA_DIR}/test.tsv \
  --output_dir=$SUB_OUTPUT_DIR --save_total_limit=50 \
  --learning_rate=${LEARN_RATE} --seed=${SEED} \
  --save_steps=4000
```
Please note that the parameter `model_name_or_path` is changed to the selected checkpoint.

Then detokenize the output and transform it to BioASQ format.
```bash
# To Detokenize the output
python detokenize-list.py \
  --test_path=${DATA_DIR}/test.tsv \
  --predictions_path=${SUB_OUTPUT_DIR}/predictions.txt \
  --original_test_path=${DATA_DIR}/original_test.json \
  --output_dir=${SUB_OUTPUT_DIR}
```

Finally, evaluate the output with official BioASQ evaluation library. Please download it from [here](https://github.com/BioASQ/Evaluation-Measures). The library requires java.
```bash
# Example script of using official BioASQ evaluation library
export EVAL_LIB=<Cloned official eval code path>/Evaluation-Measures
java -Xmx10G -cp $CLASSPATH:$EVAL_LIB/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 \
  $HOME/DATA/BioASQ-golden/${VERSION}B_total_golden.json ${SUB_OUTPUT_DIR}/NER_result_BioASQ.json >> $OUTPUT_DIR/total_BioASQ_eval.log
```

-----

For inquiries, please contact `wjyoon (_at_) korea.ac.kr`.
