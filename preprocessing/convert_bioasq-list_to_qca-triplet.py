#!/usr/bin/env python
# coding=utf-8
# Code by Wonjin Yoon for "Sequence Tagging for Biomedical Extractive Question Answering"

import json
import datetime
import os
import argparse

from utils import unnesting_list, normalize_unicode, find_all_str 
# find_all_str: 20201030 version


"""
Example of CLI script:
python convert_bioasq-list_to_qca-triplet.py \
 --input_path ../../BioASQ-original/10B1_golden.json \
 --resource_path ../../resources/pubmedDict.json \
 --output_path outputs/cqa-test.json \
 --test
"""

parser = argparse.ArgumentParser(description='')
parser.add_argument('--input_path', type=str, help="Path to BioASQ datsets ex) 'training10b.json' or '10B1_golden.json'")
parser.add_argument('--resource_path', type=str,  default="resources/pubmedDict.json", 
                    help="A dictionary with PubMed articles: key = PMID, value = {'title': '...', 'abstract': '...'}")
parser.add_argument('--output_path', type=str,  help="full path to output the file. ex) 'outputs/train.tsv'. .idx will be automatically generated.")
parser.add_argument('--question_type', type=str,  default="list")
parser.add_argument("--test", action='store_true')
parser.add_argument("--test_without_answer", action='store_true',
                    help="Set True for challenge, False for making test.tsv with answers in it.")
parser.add_argument("--debug", action='store_true')
args = parser.parse_args()

RESOURCE_PATH = args.resource_path
INPUT_PATH = args.input_path
OUTPUT_PATH = args.output_path
test=args.test
testNoAnswer=args.test_without_answer 
debug=args.debug
question_type=args.question_type

processedDate = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d-%H:%M:%S-KST")

print("question_type:", question_type)

# Read PMID-abstract file
with open(RESOURCE_PATH, 'rb') as f:
    pubmedDict=json.load(f)   
falsePmidList = ['23890950','23104528', '27399411','27399455','25064957','23986914',
                 '24305403','24305278', '27742610','23304742','26989023']

# Read bioasq file
with open(INPUT_PATH) as fp:
    raw_input_data = json.load(fp)["questions"]
print(len(raw_input_data))
# train : 2251

countDict = {"qas":0, "nopmid":0, "doc":0, 'noAnsInContext':0,'noAnsInQuestion':0}

paraResults=[]
statDict=dict() # key : qid

if True:
    # find matching Answer-Doc list
    for idx, paragraph in enumerate(raw_input_data): # iter every question.
        if paragraph[u'type']!=question_type:
            continue

        if test and testNoAnswer: # should not check matching nor exact_answer.
            exactAnswerList=[] # matches with every sentences
            #raise # not for AZ alt version now (2020-03-01) # Okay for BioASQ 9b
            pass
        else:
            exactAnswerListTmp = unnesting_list(paragraph[u'exact_answer'])
            exactAnswerList = list(filter(None, exactAnswerListTmp)) # get rid of "" (empty string)
            exactAnswerList = [normalize_unicode(ele) for ele in exactAnswerList] # normalize answers
            if len(exactAnswerList)==0 and not test:
                print("FATAL : No exact answer in traindata! \n####Abort!")
                raise
        
        docSet=set()
        cqaList=[]
        statList=list() # count answers in the paragraph
        
        # pre-check "matching" by using snippets
        pmidSet = set()
        for snippet in paragraph[u'snippets']:
            pmid=snippet[u'document'].split('/')[-1]
            
            qbody = normalize_unicode(paragraph['body']) # normalize questions (body)
            assert len(qbody) != 0
            
            if pmid in falsePmidList:
                countDict["nopmid"] += 1
                continue
            assert pmid in pubmedDict, "PMID %s not in dict"%pmid
            docSet.add(pmid)
            
        for pmid in docSet:
            occurrenceDict = dict()
            answerPerContext = 0
            
            context = pubmedDict[pmid][u'title'].strip().strip(".")+". "+pubmedDict[pmid][u'abstract'].strip().strip(".")+"."
            context = normalize_unicode(context) # normalize contexts
            
            if len(exactAnswerList) != 0: 
                for exactAnswer in set(exactAnswerList):
                    startList = list(find_all_str(exactAnswer, context, casing = "Intelli"))
                    occurrenceDict[exactAnswer] = startList
                    answerPerContext += len(startList)
                
                if answerPerContext == 0:
                    countDict["noAnsInContext"] += 1
                    if not test:
                        continue
            elif len(exactAnswerList) == 0 and test:
                startList = []
                if not testNoAnswer:
                    occurrenceDict[exactAnswer] = startList
                else:
                    pass
                countDict["noAnsInContext"] += 1
                
            else: # train and no answer in the context
                print("[Warning]: train and no answer in the context. Ignoring this sample")
                continue
            
            cqaList.append({'qid':paragraph['id'],
                            'uniqueid':paragraph['id']+"_"+("%d"%len(cqaList)).zfill(4),
                            'pmid': pmid,
                            'type': question_type,
                            'body': qbody,
                            'answerDict': {key: values for key,values in occurrenceDict.items() if len(values) != 0},
                            'nonOccurAns': [key for key,values in occurrenceDict.items() if len(values) == 0],
                            'answerPerContext': answerPerContext,
                            'context': context
                           })
            countDict["doc"] += 1
            statList.append(answerPerContext)
            
        statDict[paragraph[u'id']] = {"statList":statList, 
                                      "len":len(statList), 
                                      "nonZeroLen": len(list(filter(lambda x: x!=0, statList))), 
                                      "sum": sum(statList)}
        if statDict[paragraph[u'id']]["nonZeroLen"] != 0:
            statDict[paragraph[u'id']]["avgNumOfAns"] = statDict[paragraph[u'id']]["sum"]/float(statDict[paragraph[u'id']]["nonZeroLen"])
        else:
            statDict[paragraph[u'id']]["avgNumOfAns"] = 0
            countDict["noAnsInQuestion"] += 1
        countDict["qas"] += 1
        
        paraResults += cqaList
        
# Sanity check
for ele in paraResults:
    answerDict = ele["answerDict"]
    for answer in answerDict:
        for idx in answerDict[answer]:
            # using lower() might be dangerous but given that this is for sanity checking purpose, shall be okay.
            assert answer.lower() == ele['context'][idx:idx+len(answer)].lower(), "%s, %s"%(answer, ele['context'][idx:idx+len(answer)])
print("Sanity check done without error!")
print(processedDate)
print(countDict)

if test:
    # Only test: original_test
    newDict=dict()
    for ele in paraResults:
        key=ele["uniqueid"]
        newDict[key] = ele["context"]

    original_json_name =  os.path.splitext(OUTPUT_PATH)[0] + "-original_test.json"
    print("original_json_name:", original_json_name)
    json.dump(obj=newDict, fp=open(original_json_name, 'w'), indent=2)
    
with open(OUTPUT_PATH, 'w') as f:
    json.dump({'data': paraResults, 'processed-date': processedDate, 'countDict':countDict}, f, indent=2)
print("Saved: ", OUTPUT_PATH)
