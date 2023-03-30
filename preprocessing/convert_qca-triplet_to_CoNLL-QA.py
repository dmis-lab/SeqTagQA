#!/usr/bin/env python
# coding=utf-8
# Code by Wonjin Yoon for "Sequence Tagging for Biomedical Extractive Question Answering"

import os
import json
import datetime
import numpy as np
import argparse

from utils import smartLower
from preprocess import json_to_sent, input_form


parser = argparse.ArgumentParser(description='')
parser.add_argument('--input_path', type=str, help="Path to CQA.json ex) 'cqa-test.json'")
parser.add_argument('--output_path', type=str,  help="full path to output the file. ex) 'outputs/train.tsv'. .idx will be automatically generated.")
parser.add_argument("--eos", action='store_true')
parser.add_argument("--debug", action='store_true')
args = parser.parse_args()

INPUT_PATH = args.input_path
OUTPUT_PATH = args.output_path
eos_flag=args.eos
debug=args.debug

NLTK = False
MAX_CHARS_WORD = 25
TODO20200423 = True
scheme = "BIO"

processedDate = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d-%H:%M:%S-KST")

with open(INPUT_PATH, 'rb') as f:
    bioasqProcessedData=json.load(f)    
print(bioasqProcessedData['processed-date'])


scheme = "BIO"  # or BIOES

countDict = {'processedContext':0, 
             'manuallyExcluded':0, 
             'longWordWarning':0, 
             'TODOinverseDict':0, 
             'TODOanswerQueueNotZero':0,
             'notMatchingWordErrorInDoc':0}
bioasqTagged = {'processedDate':processedDate,'data':{}} # data : dict, key = uniqueID

for contextRaw in bioasqProcessedData['data']: # doc level
    context = contextRaw['context']
    
    # manual exception control
    if contextRaw['qid'] == "5c61bd04e842deac67000002":
        if '\u25bc' in context:
            countDict['manuallyExcluded'] += 1
            continue
            
    # transforming to match json_to_sent func
    data = {"pmid":contextRaw['uniqueid'], 
    "title":"", 
    "abstract": context}
    sentData = json_to_sent([data], is_raw_text=True)
    contextStructure = input_form(sentData, max_input_chars_per_word = MAX_CHARS_WORD)[contextRaw['uniqueid']]
    contextStructure['question'] = contextRaw['body'].splitlines()[0]  # already normalized (spacing..)
    contextStructure['tag'] = []
    
    # Sanity Check 
    longWordWarningInDoc=0
    for wordList, wordPosList in zip(contextStructure['words'], contextStructure['wordPos']): # sentence level
        # wordPos is a char-level position of !original! document. 
        for wordEle, wordPosEle in zip(wordList, wordPosList): # word level
            if context[wordPosEle[0]:wordPosEle[1]+1] != wordEle:
                print("%s, %s, %s"%(wordEle, context[wordPosEle[0]:wordPosEle[1]+1], wordPosEle))
                # ex) pharmacoep, pharmacoepidemiologic, (289, 309)
                longWordWarningInDoc += 1
    assert longWordWarningInDoc <10, "Too many longWordWarning in a context"
    countDict['longWordWarning'] += longWordWarningInDoc
    
    """ 
    Section : annotating answers 
    """
    # inverse dictionary
    inverseAnsDict = dict()
    answerList = list(contextRaw['answerDict'].keys())
    for answer in contextRaw['answerDict']:
        for pos in contextRaw['answerDict'][answer]:
            if pos in inverseAnsDict:
                if debug:
                    print("[%s]pos %s in duplicated in inverseAnsDict (answer: %s, mathching: %s)"%(contextRaw['uniqueid'], pos, answer, inverseAnsDict[pos])) # should be unique! 
                if smartLower(answer) in smartLower(inverseAnsDict[pos]): 
                    # existing word is more specific term (ex BRCA<BRCA1)
                    continue
                elif smartLower(inverseAnsDict[pos]) in smartLower(answer):
                    print("\treplacing %s with %s"%(inverseAnsDict[pos], answer))
                    pass
                else:
                    if TODO20200423:
                        countDict['TODOinverseDict'] += 1 # TODO : 2020-04-23
                    else:
                        print(inverseAnsDict[pos], answer)
                        assert 1==0, ("Duplicated pos with wrong word") 
            inverseAnsDict[pos] = answer
            
    # index-word 
    notMatchingWordErrorInDoc=0
    for wordList, wordPosList in zip(contextStructure['words'], contextStructure['wordPos']): # sentence level
        # wordPos is a char-level position of !original! document. 
        tagList = [] # sentence level
        answerQueue = [] # answer toggle/queue(by token) = false for start of an sentence
        for wordEle, wordPosEle in zip(wordList, wordPosList): # word level
            # init for a word
            tag = None
            
            # start of a entity
            if wordPosEle[0] in inverseAnsDict: 
                # checking answer span
                """
                assert len(answerQueue) == 0, f"## FATAL Error: answerSpan not empty : \n\
                   Current Answer: {answerString} \n\
                   New Answer: {inverseAnsDict[wordPosEle[0]]} \n\
                   in id: {contextRaw['uniqueid']}" 
                =>
                AssertionError: ## FATAL Error: answerSpan not empty : 
                   Current Answer: heparin-binding epidermal growth factor 
                   New Answer: epidermal growth factor 
                   in id: 55046d5ff8aee20f27000007_0003
                """
                if len(answerQueue) != 0: # position in the middle of entity
                    if inverseAnsDict[wordPosEle[0]].lower() in answerString.lower(): 
                        # lower is okay in this case (just a answer check.)
                        # Used lower to handle :
                        #  => Current Answer: CD4 and CD8 T cells | New Answer: T cells 
                        pass # should skip (not set a begin tag. It should be Intermediate)
                    else:
                        print(f"## FATAL Error: answerSpan not empty : \n\
                   Current Answer: {answerString} \n\
                   New Answer: {inverseAnsDict[wordPosEle[0]]} \n\
                   in id: {contextRaw['uniqueid']}" )
                        if TODO20200423: # TODO : 2020-04-23
                            countDict['TODOanswerQueueNotZero'] += 1 # TODO : 2020-04-23
                            answerQueue = []
                        else:
                            raise
                else: # start of answer
                    answerString = inverseAnsDict[wordPosEle[0]]
                    answerJson = json_to_sent(data = [{"pmid":"dummypmid", 
                                                      "title":"", 
                                                      "abstract": answerString}], is_raw_text=True)
                    answerInpForm = input_form(answerJson, 
                                               max_input_chars_per_word = MAX_CHARS_WORD)["dummypmid"]
                    
                    answerQueue = answerInpForm['words'][0]
                        
                    #answerPosQueue = answerInpForm['wordPos'][0]
                    #print(answerQueue)
                    tag = "Begin"
            
            # set tag if the word is 
            if len(answerQueue) != 0:
                # word level matching checking
                if smartLower(answerQueue[0])[:MAX_CHARS_WORD] != smartLower(context[wordPosEle[0]:wordPosEle[1]+1])[:MAX_CHARS_WORD]: # olny compare first 10 char
                    if answerQueue[0].lower()[:MAX_CHARS_WORD] == context[wordPosEle[0]:wordPosEle[1]+1][:MAX_CHARS_WORD].lower():
                        # is okay. Just casing problem
                        pass 
                    else:
                        print(f"Answer String : {answerString}\n\
answer tok at now pos:\t %s \n\
answer tok from context: %s \n\
Position: %s\n\
id: {contextRaw['uniqueid']}"%(answerQueue[0], context[wordPosEle[0]:wordPosEle[1]+1], wordPosEle))
                        notMatchingWordErrorInDoc += 1 # TODO : change this to assert                    
                        raise
                        
                if tag == None:
                    tag = "Intermediate"
                del answerQueue[0] # reverse pop
                #del answerPosQueue[0]
                if len(answerQueue) == 0:
                    if tag != "Begin":
                        tag = "End"
            
            if tag == None:
                tag = "Out" 
            
            if scheme == "BIO":
                tag = tag.replace("End", "I")
                tagList.append(tag[0])
            else:
                raise
        assert len(tagList) == len(wordList), "len tagList not matching with wordList"
        contextStructure['tag'].append(tagList)
        
    assert len(answerQueue) == 0, "FATAL Error: answerSpan not empty after sentence ends."
    assert notMatchingWordErrorInDoc <10, "Too many notMatchingWordError in a context"
    
    countDict['notMatchingWordErrorInDoc'] += notMatchingWordErrorInDoc    
    assert len(contextStructure['tag']) == len(contextStructure['words'])
    bioasqTagged['data'][contextRaw['uniqueid']] = {'question': contextStructure['question'],
                                                    'words': contextStructure['words'], 
                                                    'wordPos': contextStructure['wordPos'],
                                                    'tag': contextStructure['tag'],
                                                    'answerDict':contextRaw['answerDict'],
                                                   }
    countDict['processedContext']+=1
    
bioasqTagged['countDict'] = countDict
print(countDict)

#######################################
#### Part II tokenization-CoNLL-QA ####
#######################################
 
idxDict = dict()
tsvFile = open(OUTPUT_PATH, 'w')

offset = 1
sentCount = 0

if True: 
    for uid, dataDict in bioasqTagged['data'].items():
        idxDict[uid] = []
        if uid == "5c990afbecadf2e73f000030_0004":
            print("printing uid: %s"%uid)
            print(" ".join([" ".join(ele) for ele in dataDict['words']]))
        tsvFile.write("UNIQUEID\t%s\t%s\n"%(uid, dataDict['question']))
        offset += 1
        
        # iter every sent
        for sent, tagSent in zip(dataDict['words'], dataDict['tag']):
            dataStartEndDict = {'start':offset}
            # iter every word
            for idx, (word, tag) in enumerate(zip(sent, tagSent)):
                tsvFile.write("%s\t%s\n"%(word, tag))
            
            # End of sentence
            if eos_flag:
                idx += 1
                tsvFile.write("%s\t%s\n"%("¶", 'O'))
            
            offset += idx + 1
            sentCount += 1
            dataStartEndDict['end'] = offset - 1
            idxDict[uid].append(dataStartEndDict)
            
        # blank space split for question
        tsvFile.write("\n")
        offset += 1
            
tsvFile.close()

with open('%s.idx'%(os.path.splitext(OUTPUT_PATH)[0]), 'w') as idxFile:
    json.dump(idxDict, idxFile, indent=2)
    print(len(idxDict), "idx written")
    
    assert sentCount == sum([len(value) for key, value in idxDict.items()]), "sentCount : %s, sum of idxDict: %s"%(sentCount, 
                                                                                                                   sum([len(value) for key, value in idxDict.items()]))
