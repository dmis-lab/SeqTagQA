# Copyright  
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

import argparse
import json
import os
import string
from typing import Dict

parser = argparse.ArgumentParser(description='')
parser.add_argument('--test_path', type=str,  help='Path to test.tsv from dataset folder. ex) dataset/test.tsv')
parser.add_argument('--predictions_path', type=str,  help='Path to predictions.txt from output folder. ex) output/predictions.txt')
parser.add_argument('--original_test_path', type=str,  help='Path to original_test.json file. ex) dataset/original_test.txt')
parser.add_argument('--output_dir', type=str,  help='Path to output result will write on. ex) output/')
parser.add_argument('--debug', action='store_true', help='Debug. Outputs NER_result_sent-debug.txt')
args = parser.parse_args()

def specialchar_norm(spe_char):
    transl_table = dict( [ (ord(x), ord(y)) for x,y in zip( u"‘’´“”–-—",  u"'''\"\"---") ] ) 
    return spe_char.translate( transl_table )

def find_cursor(t, subString, uniqueID="NotGiven", debug=False) -> int:
    """
    (Recursive) function to find the char-level position of token t in the given subString. Used to map tokenized word to original string
    t: token
    subString: subString of the original string from original_test.json
    uniqueID: (Optional) uniqueID for debugging

    return : [int] the position of t in the given subString
    """


    if t == '¶': # END of a sentence (inserted during pre-processing steps)
        # Do not modify currentPosition as it is not included in the original string. 
        return 0
    else:
        assert subString != ""
        
    if subString[0] in [" ", '\u200b','\n', '\ufeff']: 
        # Starts with a whitespace or return char (erronius original sample) or BOM marker
        return 1 + find_cursor(t, subString[1:], uniqueID=uniqueID, debug=debug) # (recursive func) Remove one WS and continue
        
    if t == subString[:len(t)]: 
        return len(t)
    elif t == specialchar_norm(subString[:len(t)]):
        if debug:
            print(f"Special char normalized: {subString[:len(t)]} -> {t}")
        return len(t)
    else: # Edge cases
        # Error caused by special char (replaced with [UNK])
        # EX) upper char of TM -> [UNK]
        if t=='[UNK]':
            if debug:
                print("\nException caused by special char. [UNK]")
                print("Now -> | Word token t: '%s' String : '%s'"%(t, subString[:40]))
            return 1

        # Case that a word (t) is truncated during re-processing due to word char restrictions (>22) 
        # EX) phenoxymethylpenicillin -> phenoxymethylpenicilli
        # EX) pneumonoultramicroscopic-silicovolcanoconiosis -> pneumonoultramicroscopi - silicovolcanoconiosis
        replacedSubString = subString
        for ele in string.punctuation:
            replacedSubString = (" %s "%ele).join(replacedSubString.split(ele)).strip().replace("  ", " ") # adding a space near special char

        if t == replacedSubString.split()[1][:len(t)]:
            if debug:
                print("\n## Word (t) truncated\nUNIQUEID: %s | subString: %s"%(uniqueID, subString))
                print("Now -> | Word token t: '%s' String : '%s'"%(t, subString[:40]))
            lenTruncatedPart = len(replacedSubString.split()[0])
            return lenTruncatedPart + find_cursor(t=t, subString=subString[lenTruncatedPart:], uniqueID=uniqueID, debug=debug) # (recursive func) Remove one WS and continue
             # remove truncated part

        else: # UNKNOWN ERROR : FATAL [SHOULD BE DEBUGED] 
            print("\n## UNKNOWN ERROR\nUNIQUEID: %s | subString: %s"%(uniqueID, subString))
            print("Now -> | Word token t: '%s' String : '%s'"%(t, subString[:40]))
            import pdb;pdb.set_trace()
            raise NotImplementedError

def _read_tokens_and_labels(test_path, predictions_path, debug) -> Dict:
    """
    internal function. Used in detokenize_uid function

    Args:
        test_path: Path to test.tsv from dataset folder. 
        predictions_path: Path to predictions.txt 
    """

    pred = {'toks':[], 'labels':[]} # dictionary for predicted tokens and labels. List of list (sample)

    with open(test_path,'r') as in_tok, open(predictions_path,'r') as in_lab: #'token_test.txt'
        tokens = []
        labels = []

        # read test dataset
        for lineIdx, lineTok in enumerate(in_tok.readlines()):
            lineTok = lineTok.splitlines()[0]
            if lineIdx != 0 and lineTok == "": # new sample
                pred['toks'].append(tokens)
                tokens = []
                continue
            else:
                tokens.append(lineTok)
        if tokens != []:
            pred['toks'].append(tokens)

        # read predicted
        for example_pred in in_lab.readlines():
            example_pred = example_pred.splitlines()[0]
            if example_pred == "":
                continue
            for label_idx, label in enumerate(example_pred.split()):
                if label_idx == 0:
                    assert label not in ["B", "I", "O"], "First predicted label of each sample should be UNIQUEID"
                labels.append(label)
            assert label != [], f"label should not be an empty list label_idx: {label_idx}"
            pred['labels'].append(labels)
            labels = []

    assert (len(pred['toks']) == len(pred['labels'])), "Error! : testdata len(pred['toks'])(%s) != output len(pred['labels'])(%s) : Please report us "%(len(pred['toks']), len(pred['labels']))
    
    return pred

def detokenize_uid(test_path, predictions_path, original_test_path, debug):
    """
    convert sub-word level BioBERT-NER results to full words, labels and 
      charactor-level whitespace mapping information between the original string and pre-processed sequence.
        
    Args:
        test_path: Path to test.tsv from dataset folder. 
        predictions_path: Path to predictions.txt 
        original_test_path: Path to original_test.json file. 
            Whitespaces are inserted near special charactors during pre-processing steps. 
            original_test.json file is used to calculate mapping information for restoring original string.
            original_test.json may not exactly same with "raw data" as some unicode charictors are normailzed. 
    Outs:
        A dictionary that contains full words and predicted labels. 
    """
    # read original strings (Strings before pre-processing; Used for restoring pre-white spaces added while processing)
    pred = _read_tokens_and_labels(test_path, predictions_path, debug)

    with open(original_test_path, "r") as originFile:
        originStringDict = json.load(originFile)

    bertPredDict = {}
    for t_example, l_example in zip(pred['toks'], pred['labels']):

        bert_pred = {'toks':[], 'labels':[], 'testdata':[], "charPos":[]}
        charPos = 0
        """
             charPos: charactor level position curser (after previous word) in original string. 
             Ideally, if t dose not start from ##, originString[charPos+1:] should start from current word.
             If t starts from ##, originString[charPos:] should start from current sub-word.
             ex) "New drug^has..." : t="drug", charPos=8, a[8:]='^has...'
        """
        uniqueID, *bert_pred['labels'] = l_example
        tmp_uniqueID, *bert_pred['testdata'] = t_example
        bert_pred['toks'] = [tok.split("\t")[0] for tok in bert_pred['testdata']][:len(bert_pred['labels'])] # some inputs are truncated: SQuAD : 65 in test

        assert uniqueID in tmp_uniqueID
        assert uniqueID in originStringDict

        originString = originStringDict[uniqueID]

        for tok in bert_pred['toks']:
            try:
                charPos = charPos + find_cursor(t=tok, subString=originString[charPos:], uniqueID=uniqueID, debug=debug)
            except NotImplementedError:
                print("[FATAL] Previous word '%s' String near t : \n'%s'"%(bert_pred['toks'][-2], originString[charPos-20:charPos+20]))
                break

            assert tok == originString[charPos-len(tok) : charPos] or \
                (tok in ['¶', "[UNK]"]) or \
                (tok == specialchar_norm(originString[charPos-len(tok) : charPos])), \
                "t=%s, charPos=%s, subString=%s"%(tok, charPos, originString[charPos-len(tok) : charPos])
            bert_pred['charPos'].append(charPos)

        bertPredDict[uniqueID] = bert_pred

    for uid, bert_pred in bertPredDict.items():
        assert (len(bert_pred['toks']) == len(bert_pred['labels'])), (f"Error! : len(bert_pred['toks']) ({len(bert_pred['toks'])}) != len(bert_pred['labels']) ({len(bert_pred['labels'])}) : Please report us")
        assert (len(bert_pred['toks']) == len(bert_pred['charPos'])), (f"Error! : len(bert_pred['toks']) != len(bert_pred['charPos']) : Please report us")
    return bertPredDict


def answer_detokenize(answerCandi, answerPosit, uniqueID="NotGiven", prevTok=None):
    """
    Revert/Detokenize answer using original string mapping infomation (answerPosit) : "c . 1516c > t" to "c.1516C>T"
    """
    assert len(answerCandi) == len(answerPosit), "len(answerCandi) != len(answerPosit) %s != %s at UID %s"%(len(answerCandi), len(answerPosit), uniqueID)
    assert len(answerPosit) >= 1 # len(answerPosit) != 0 

    if answerCandi[0] == "[UNK]":
        print("WARNING: [UNK] in answer. UID : %s"%uniqueID)
        answerCandi[0] = "U"
    elif answerCandi[0] == "¶":
        print("WARNING: EOS (¶) in answer. UID : %s"%uniqueID)
        answerCandi[0] = ""

    if prevTok == None: # the first iter of the recursive function
        outString = answerCandi[0]
    # it there are precedent token(s) 
    elif answerPosit[0] - prevTok == len(answerCandi[0]):
        outString = answerCandi[0]
    elif answerPosit[0] - prevTok == len(answerCandi[0]) + 1: # Whitespace / precedent token is truncated 
        outString = " " + answerCandi[0]
    elif answerPosit[0] - prevTok > len(answerCandi[0]) + 1: # precedent token is truncated 
        print("[WARNING][TODO] Check the input. Truncated at %s:",uniqueID)
        #pdb.set_trace()
        outString = " " + answerCandi[0]
    else:
        raise ValueError(f"Unexpected error at uniqueID:{uniqueID}")
    
    if len(answerPosit) == 1:
        return outString 
    elif len(answerPosit) >= 2:
        return outString + answer_detokenize(answerCandi[1:], answerPosit[1:], uniqueID=uniqueID, prevTok=answerPosit[0]) 


def transform2BERTQA(output_dir, bertPredDict, debug):
    """
    Produce NER_result_BioASQ.json file that suits BioASQ official eval script.
    No need for golden
    Output : List of dictionaries that has id and multiple answer candidates.
    ...
    """
    BioASQDictListRaw = []
    for uid, bert_pred in bertPredDict.items():
        BioASQDict = {'unique_id':uid, 'qid':uid.split('_')[0], 'exact_answer':[], 'exact_answer_tmp':[] }
        # 'exact_answer': list of lists that has an answer  'exact_answer_tmp':[] list of answers
        answerCandi = []
        answerPosit = []
        for toks, labels, charPos in zip(bert_pred['toks'], bert_pred['labels'], bert_pred['charPos']):
            # BIO setting
            if labels != 'I':
                # append current answer candidate into exact answer
                if len(answerCandi) != 0:
                    BioASQDict['exact_answer_tmp'].append(answer_detokenize(answerCandi=answerCandi, answerPosit=answerPosit, uniqueID=uid, prevTok=None))#(" ".join(answerCandi))
                    answerCandi = [] # reset
                    answerPosit = []
                    
            if labels in ['B', 'I']:
                answerCandi.append(toks)
                answerPosit.append(charPos)
        if len(answerCandi) != 0:
            BioASQDict['exact_answer_tmp'].append(answer_detokenize(answerCandi=answerCandi, answerPosit=answerPosit, uniqueID=uid, prevTok=None))#(" ".join(answerCandi))
            answerCandi = [] # reset
            answerPosit = []

        # make answers unique ; lower cased match -> not a graceful way but it works!
        BioASQDict['exact_answer_tmp'] = list(set([ele.lower() for ele in BioASQDict['exact_answer_tmp']]))
        BioASQDict['exact_answer'] = [[ele] for ele in BioASQDict['exact_answer_tmp']]

        BioASQDictListRaw.append(BioASQDict)
    
    # Output
    json.dump({"questions":BioASQDictListRaw }, 
              open(os.path.join(output_dir, "NER_result_BioASQ-raw.json"), "w"),
              indent=2)
   
    
    # merging with qid
    qidDict = dict()
    for ele in BioASQDictListRaw:
        if ele['qid'] in qidDict:
            try:
                qidDict[ele['qid']] = list(set(qidDict[ele['qid']] + ele['exact_answer_tmp']))
            except:
                print("ele['qid'] : ", ele['qid'])
                print("qidDict[ele['qid']] : ", qidDict[ele['qid']])
                print("ele['exact_answer_tmp'] : ", ele['exact_answer_tmp'])
        else:
            qidDict[ele['qid']] = ele['exact_answer_tmp']
    
    returnDict = dict()
    for key, value in qidDict.items():
        if len(value) == 0:
            returnDict[key] = ''
        else:
            returnDict[key] = value[0]
    # Output
    json.dump(returnDict, 
              open(os.path.join(output_dir, "NER_result_BERTQA_predictions.json"), "w"),
              indent=2)


def transform2BioASQ(output_dir, bertPredDict, debug):
    """
    Produce NER_result_BioASQ.json file that suits BioASQ official eval script.
    No need for golden
    Output : List of dictionaries that has id and multiple answer candidates.
    ...
    """

    BioASQDictListRaw = []
    for uid, bert_pred in bertPredDict.items():
        BioASQDict = {'unique_id':uid, 'qid':uid.split('_')[0], 'exact_answer':[], 'exact_answer_tmp':[] }
        # 'exact_answer': list of lists that has an answer  'exact_answer_tmp':[] list of answers
        answerCandi = []
        answerPosit = []
        for toks, labels, charPos in zip(bert_pred['toks'], bert_pred['labels'], bert_pred['charPos']):
            # BIO setting
            if labels != 'I':
                # append current answer candidate into exact answer
                if len(answerCandi) != 0:
                    BioASQDict['exact_answer_tmp'].append(answer_detokenize(answerCandi=answerCandi, answerPosit=answerPosit, uniqueID=uid, prevTok=None))
                    answerCandi = [] # reset
                    answerPosit = []
                    
            if labels in ['B', 'I']:
                answerCandi.append(toks)
                answerPosit.append(charPos)
        if len(answerCandi) != 0:
            BioASQDict['exact_answer_tmp'].append(answer_detokenize(answerCandi=answerCandi, answerPosit=answerPosit, uniqueID=uid, prevTok=None))
            answerCandi = [] # reset
            answerPosit = []

        # make answers unique ; lower cased match -> not a graceful way but it works!
        BioASQDict['exact_answer_tmp'] = list(set([ele.lower() for ele in BioASQDict['exact_answer_tmp']]))
        BioASQDict['exact_answer'] = [[ele] for ele in BioASQDict['exact_answer_tmp']]

        BioASQDictListRaw.append(BioASQDict)
    
    # Output
    json.dump({"questions":BioASQDictListRaw }, 
              open(os.path.join(output_dir, "NER_result_BioASQ-raw.json"), "w"),
              indent=2)
   
    
    # merging with qid
    qidDict = dict()
    for ele in BioASQDictListRaw:
        if ele['qid'] in qidDict:
            try:
                qidDict[ele['qid']] = list(set(qidDict[ele['qid']] + ele['exact_answer_tmp']))
            except:
                print("ele['qid'] : ", ele['qid'])
                print("qidDict[ele['qid']] : ", qidDict[ele['qid']])
                print("ele['exact_answer_tmp'] : ", ele['exact_answer_tmp'])
        else:
            qidDict[ele['qid']] = ele['exact_answer_tmp']
    
    BioASQDictList = []
    for key, ele in qidDict.items():
        ele = [[answer] for answer in ele]
        BioASQDictList.append({'id':key, 'type':"list", 'exact_answer':ele})
    
    # Output
    json.dump({"questions":BioASQDictList }, 
              open(os.path.join(output_dir, "NER_result_BioASQ.json"), "w"),
              indent=2)

    return BioASQDictList


if __name__ == "__main__":
    bertPredDict = detokenize_uid(test_path=args.test_path, predictions_path=args.predictions_path, original_test_path=args.original_test_path, debug=args.debug)
    # with open(os.path.join(args.output_dir, "bertPredDict.json"), "w") as bertPredDictFile:
    #     json.dump(obj=bertPredDict, fp=bertPredDictFile, indent=2)
    transform2BioASQ(output_dir=args.output_dir, bertPredDict=bertPredDict, debug=args.debug)
    transform2BERTQA(output_dir=args.output_dir, bertPredDict=bertPredDict, debug=args.debug)
