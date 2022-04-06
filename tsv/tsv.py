#!/usr/bin/env python
# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom dataset loading script for CoNLL-like QA (STQA: Sequence Taggging Question Answering) dataset (tsv)"""

import os
import datasets
from typing import List

logger = datasets.logging.get_logger(__name__)


_CITATION = """Yoon et al."""
_DESCRIPTION = """Dataset builder for QA dataset in STQA format"""

_TRAINING_FILE = "train_dev.tsv"
_TEST_FILE = "test.tsv"
_DEV_FILE = _TEST_FILE


class SeqTagQuestionAnsweringConfig(datasets.BuilderConfig):
    """BuilderConfig for STQA"""

    def __init__(self, **kwargs):
        """BuilderConfig for conllqa.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SeqTagQuestionAnsweringConfig, self).__init__(**kwargs)


class SeqTagQuestionAnsweringDatasetBuilder(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        SeqTagQuestionAnsweringConfig(name="stqa", version=datasets.Version("1.0.0"), description="stqa dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "unique_id": datasets.Value("string"),
                    "context_tokens": datasets.Sequence(datasets.Value("string")),
                    "question_raw": datasets.Value("string"), # for debug 
                    "question": datasets.Sequence(datasets.Value("string")),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "answer_labels": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "[PAD]", "B", "I", "O", "X", "[CLS]", "[SEP]"
                            ]
                        )
                    ),
                    "labels": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["[PAD]", "B", "I", "O", "X", "[CLS]", "[SEP]"]
                    )),
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """Returns SplitGenerators."""
        
        data_files = dict()
        for data_split_type, data_path in self.config.data_files.items():
            data_files[data_split_type] = data_path[0] if type(data_path) == datasets.data_files.DataFilesList else data_path

        generator_list = []
        if "train" in data_files:
            generator_list.append(datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["train"]}))
        if "validation" in data_files:
            generator_list.append(datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": data_files["validation"]}))
        if "test" in data_files:
            generator_list.append(datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"]}))
        
        return generator_list

    def _question_tokenizer(self, question_raw):
        question = question_raw.replace("  ", " ")

        specialChar = ".,:;()/!?'$`\\" + '"'
        for char in specialChar:
            question = question.replace(char, " %s "%char)
        question = question.replace("  ", " ").replace("  ", " ")
        return question.split()


    def _generate_examples(self, filepath):
        logger.info("Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            unique_id = ""
            context_tokens = []
            question_raw = ""
            question_tokens = []
            tokens = []
            answer_labels = []
            labels = []
            for line in f:
                if line.startswith("UNIQUEID") or line == "" or line == "\n":
                    if line.startswith("UNIQUEID"):
                        splits = line.splitlines()[0].split("\t")
                        unique_id = splits[1]
                        question_raw = splits[2]

                        question_tokens = self._question_tokenizer(question_raw)
                    
                    if context_tokens:
                        tokens = question_tokens + context_tokens
                        # labels overwritten in tokenize_and_align_labels_pair function
                        labels = ["X"] * len(question_tokens) + answer_labels
                        assert len(tokens) == len(labels)

                        yield guid, {
                            "id": str(guid),
                            "unique_id": unique_id,
                            "context_tokens": context_tokens,
                            "question_raw": question_raw,
                            "question": question_tokens,
                            "tokens": tokens, # TODO : remove this
                            "answer_labels": answer_labels,
                            "labels": labels,
                        }
                        guid += 1
                        unique_id = ""
                        context_tokens = []
                        question_raw = ""
                        question_tokens = []
                        tokens = []
                        answer_labels = []
                        labels = []
                else:
                    # conll2003 tokens are space separated
                    splits = line.splitlines()[0].split("\t")
                    context_tokens.append(splits[0])
                    answer_labels.append(splits[1])

            # last example
            yield guid, {
                "id": str(guid),
                "unique_id": unique_id,
                "context_tokens": context_tokens,
                "question_raw": question_raw,
                "question": question_tokens,
                "tokens": tokens,
                "answer_labels": answer_labels,
                "labels": labels,
            }

