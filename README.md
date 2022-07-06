# Sequence Tagging for Biomedical Extractive Question Answering

Source codes and resources for "Sequence Tagging for Biomedical Extractive Question Answering."

Please cite:
 ```bib
 @article{yoon-etal-2022-sequence,
    author = {Yoon, Wonjin and Jackson, Richard and Lagerberg, Aron and Kang, Jaewoo},
    title = "{Sequence tagging for biomedical extractive question answering}",
    journal = {Bioinformatics},
    year = {2022},
    month = {06},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btac397},
}
 ```
```
Wonjin Yoon, Richard Jackson, Aron Lagerberg, Jaewoo Kang, Sequence tagging for biomedical extractive question answering, Bioinformatics, 2022;, btac397, https://doi.org/10.1093/bioinformatics/btac397
```
## _Naturally posed_ biomedical questions
In the paper, we investigated the characteristics of naturally posed biomedical questions as a preliminary study. 
As a wish to explore the nature of biomedical question answering with the community, we made the collection of biomedical questions available for download in this repository.

[**See the online preview of the questions here**](https://1drv.ms/x/s!AjwviG8mocn7hbp16IRH8ICJQKcTXw?e=uYwXiV)

The questions are from _Clinical Questions Collection (CQC)_ (D’Alessandro et al., 2004; Ely et al., 1999, 1997) and _PubMed queries_ (Herskovic et al., 2007), with **our _query screening algorithm_ (Please check [Section 3 of our paper](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btac397/6609766#364116020) and [appendix](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/btac397/6609766#supplementary-data)).** 

From our analysis in the sampled question subset, (where an equal number of questions are randomly sampled from two sources of questions), about half of the questions are "Others", which means they are not answerable with current settings. 
The resons can be various including: 
* One or more conditions are missing: (as suggested in the [ConditionalQA (Sun et al., 2021)](https://arxiv.org/abs/2110.06884))
* The question is too abstractive or requires a long answer that cannot be extracted from a paragraph: (ex. "what is the history of calcium?" or "what can i do?") 
* Incompleteness in our query screening algorithm: (ex. "who diagnostic criteria" or "Will and Joondeph")

#### I am interested in exploring the characteristics of biomedical questions or building QA datasets of naturally posed biomedical questions. (As a future work)
The work on the characteristics of naturally posed biomedical questions in this paper is a preliminary study and I think there is huge room for improvement. 
Any researchers who want to collaborate with me on this topic are welcome! To discuss, please contact me: `wonjin.info (_at_) gmail.com`. If you are planning to attend the 10th BioASQ (2022), we can discuss it there!

-----

## How to train/evaluate

For datasets (including biomedical questions), please download them from: [here](https://drive.google.com/file/d/1m0GnVlKqvUHfDdpZ9KDDor5EiIqhPwp3/view?usp=sharing)
<br>To use the BioASQ dataset, you need to register in [the BioASQ website](http://participants-area.bioasq.org/general_information/general_information_registration/) which authorizes the use of the dataset. 

For example training and evaluation bash script: please see [`train_list_example.sh`](./train_list_example.sh)

### Detailed instructions
#### Training
The following instructions are example scripts for training LM on the BioASQ8b dataset.

In the first part, we set environmental variables.
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
Then select the latest checkpoint and predict the test dataset using the checkpoint.
```bash
export CKPT=84000
export SUB_OUTPUT_DIR=${OUTPUT_DIR}/checkpoint-${CKPT}

python run_eqa.py --model_struct=linear \
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

Finally, evaluate the output with the official BioASQ evaluation library. Please download it from [here](https://github.com/BioASQ/Evaluation-Measures). The library requires java.
```bash
# Example script of using official BioASQ evaluation library
export EVAL_LIB=<Cloned official eval code path>/Evaluation-Measures
java -Xmx10G -cp $CLASSPATH:$EVAL_LIB/flat/BioASQEvaluation/dist/BioASQEvaluation.jar evaluation.EvaluatorTask1b -phaseB -e 5 \
  $HOME/DATA/BioASQ-golden/${VERSION}B_total_golden.json ${SUB_OUTPUT_DIR}/NER_result_BioASQ.json >> $OUTPUT_DIR/total_BioASQ_eval.log
```

-----

For inquiries, please contact `wonjin.info (_at_) gmail.com`.
