#
# Original code from https://github.com/dmis-lab/bern/blob/master/biobert_ner/ops.py
# Modified by Wonjin Yoon (wonjin.info) for BioBERT SeqTag task
#

import numpy as np
import re


NLTK = False
tokenize_regex = re.compile(r'([0-9a-zA-Z]+|[^0-9a-zA-Z])')



def json_to_sent(data, is_raw_text=False):
    '''data: list of json file [{pmid,abstract,title}, ...] '''
    out = dict()
    for paper in data:
        sentences = list()
        if is_raw_text:
            # assure that paper['abstract'] is not empty
            abst = sentence_split(paper['abstract'])
            if len(abst) != 1 or len(abst[0].strip()) > 0:
                sentences.extend(abst)
        else:
            # assure that paper['title'] is not empty
            if len(CoNLL_tokenizer(paper['title'])) < 50:
                title = [paper['title']]
            else:
                title = sentence_split(paper['title'])
            if len(title) != 1 or len(title[0].strip()) > 0:
                sentences.extend(title)

            if len(paper['abstract']) > 0:
                abst = sentence_split(' ' + paper['abstract'])
                    
                if len(abst) != 1 or len(abst[0].strip()) > 0:
                    sentences.extend(abst)

        out[paper['pmid']] = dict()
        out[paper['pmid']]['sentence'] = sentences
    return out


def input_form(sent_data, max_input_chars_per_word=20):
    '''sent_data: dict of sentence, key=pmid {pmid:[sent,sent, ...], pmid: ...}'''
    for pmid in sent_data:
        sent_data[pmid]['words'] = list()
        sent_data[pmid]['wordPos'] = list()
        doc_piv = 0
        for sent in sent_data[pmid]['sentence']:
            wids = list()
            wpos = list()
            sent_piv = 0
            tok = CoNLL_tokenizer(sent)

            for w in tok:
                if len(w) > max_input_chars_per_word: # was 20
                    wids.append(w[:max_input_chars_per_word]) # was 10
                else:
                    wids.append(w)

                start = doc_piv + sent_piv + sent[sent_piv:].find(w)
                end = start + len(w) - 1
                sent_piv = end - doc_piv + 1
                wpos.append((start, end))
            doc_piv += len(sent)
            sent_data[pmid]['words'].append(wids)
            sent_data[pmid]['wordPos'].append(wpos)

    return sent_data


def isInt(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


def isFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def softmax(logits):
    out = list()
    for logit in logits:
        temp = np.subtract(logit, np.max(logit))
        p = np.exp(temp) / np.sum(np.exp(temp))
        out.append(np.max(p))
    return out


def CoNLL_tokenizer(text):
    rawTok = [t for t in tokenize_regex.split(text) if t]
    assert ''.join(rawTok) == text
    tok = [t for t in rawTok if t != ' ']
    return tok


def sentence_split(text):
    sentences = list()
    sent = ''
    piv = 0
    for idx, char in enumerate(text):
        if char in "?!":
            if idx > len(text) - 3:
                sent = text[piv:]
                piv = -1
            else:
                sent = text[piv:idx + 1]
                piv = idx + 1

        elif char == '.':
            if idx > len(text) - 3:
                sent = text[piv:]
                piv = -1
            elif (text[idx + 1] == ' ') and (
                    text[idx + 2] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ-"' + "'"):
                sent = text[piv:idx + 1]
                piv = idx + 1

        if sent != '':
            toks = CoNLL_tokenizer(sent)
            if len(toks) > 100:
                while True:
                    rawTok = [t for t in tokenize_regex.split(sent) if t]
                    cut = ''.join(rawTok[:200])
                    sent = ''.join(rawTok[200:])
                    sentences.append(cut)

                    if len(CoNLL_tokenizer(sent)) < 100:
                        if sent.strip() == '':
                            sent = ''
                            break
                        else:
                            sentences.append(sent)
                            sent = ''
                            break
            else:
                sentences.append(sent)
                sent = ''

            if piv == -1:
                break

    if piv != -1:
        sent = text[piv:]
        toks = CoNLL_tokenizer(sent)
        if len(toks) > 100:
            while True:
                rawTok = [t for t in tokenize_regex.split(sent) if t]
                cut = ''.join(rawTok[:200])
                sent = ''.join(rawTok[200:])
                sentences.append(cut)

                if len(CoNLL_tokenizer(sent)) < 100:
                    if sent.strip() == '':
                        sent = ''
                        break
                    else:
                        sentences.append(sent)
                        sent = ''
                        break
        else:
            sentences.append(sent)
            sent = ''

    return sentences


def detokenize(tokens, predicts, logits):
    pred = dict({
        'toks': tokens[:],
        'labels': predicts[:],
        'logits': logits[:]
    })  # dictionary for predicted tokens and labels.

    bert_toks = list()
    bert_labels = list()
    bert_logits = list()
    tmp_p = list()
    tmp_l = list()
    tmp_s = list()
    for t, l, s in zip(pred['toks'], pred['labels'], pred['logits']):
        if t == '[CLS]':  # non-text tokens will not be evaluated.
            continue
        elif t == '[SEP]':  # newline
            bert_toks.append(tmp_p)
            bert_labels.append(tmp_l)
            bert_logits.append(tmp_s)
            tmp_p = list()
            tmp_l = list()
            tmp_s = list()
            continue
        elif t[:2] == '##':  # if it is a piece of a word (broken by Word Piece tokenizer)
            tmp_p[-1] = tmp_p[-1] + t[2:]  # append pieces
        else:
            tmp_p.append(t)
            tmp_l.append(l)
            tmp_s.append(s)

    return bert_toks, bert_labels, bert_logits
