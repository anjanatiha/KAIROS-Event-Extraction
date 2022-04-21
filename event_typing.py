# import os
import json
import re
from operator import itemgetter

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import words
from nltk.corpus import wordnet as wn
from nltk.corpus import framenet as fn
from nltk.corpus import stopwords
from nltk.corpus.reader.framenet import PrettyList
from nltk.stem.snowball import SnowballStemmer
from time import time

nltk.download('words')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

def get_event_typing(w, thres = 0.0, m = 3, depth = 3):
    ws = wn.synsets(w)
    
    hyper = lambda s: s.hypernyms()
    types1 = {}
    types2 = {}

    for e in ws:
        w_tmp = e.name().split(".")[0]
        if w_tmp==w:
            t = 0
            c = 0 
            i = 0
            tmp_types = ""
            
            tmp_event_types = list(e.closure(hyper))
            
            for tmp_event in tmp_event_types:
                d = wn.path_similarity(e, tmp_event)
                tmp_event_name = tmp_event.name().split(".")[0]
                
                b = m**i
                t = t + (d/b)
                types2[tmp_event_name] = d
                
                if d and (d >= (thres/t)):
                    tmp_types = tmp_types + ":" + tmp_event_name
                    if c >= depth - 1:
                        break
                    c = c + 1
                    
                i = i + 1
                
            if len(tmp_types) > 0:
                types1[tmp_types] = t
                
    types1 = {k: v for k, v in sorted(types1.items(), key=lambda item: item[1], reverse=True)}
    types2 = {k: v for k, v in sorted(types2.items(), key=lambda item: item[1], reverse=True)}
    
    return types1, types2


def get_event_typing_helper(w, pos, thres = 0.0, m = 3, depth = 3):
    if not pos:
        pos = nltk.tag.pos_tag(w.split())[0][1]
    
    types1, types2 = get_event_typing(w, thres, m, depth)
    
    if not (types1 or types2):
        if pos.startswith("V"):    
            w_tmp = WordNetLemmatizer().lemmatize(w, "v")
            types1, types2 = get_event_typing(w_tmp, thres, m, depth)
            if (types1 or types2):
                return types1, types2
            
        if pos.startswith("N"):
            w_tmp = WordNetLemmatizer().lemmatize(w, "n")
            types1, types2 = get_event_typing(w_tmp, thres, m, depth)
            if (types1 or types2):
                return types1, types2
            
        w_tmp = WordNetLemmatizer().lemmatize(w)
        types1, types2 = get_event_typing(w_tmp, thres, m, depth)
        
        if (types1 or types2):
            return types1, types2
        
        stemmer = SnowballStemmer("english")
        w_tmp = stemmer.stem(w)
        types1, types2 = get_event_typing(w_tmp, thres, m, depth)

    return types1, types2

def get_event_typing_main(w, pos, thres = 0.0, m = 3, depth = 3):
    types1, types2 = get_event_typing_helper(w, pos, thres = 0.0, m = 3, depth = 3)

    if types1:
        return list(types1.keys())[0]
    else:
        return list(types2.keys())[0]



