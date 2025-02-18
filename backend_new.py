import cherrypy
import cherrypy_cors
import json
import os
from util_new import *
import re
from time import time
import regex

# import < your_code >

from nltk import sent_tokenize, word_tokenize

special_char_list = ["'", ",", ";", ":", "-", "?", "!", "$", "%", "#", "_", "&", "~", "|", "^", "+", "*", "/", "<", "=", ">", "(", ")", "{", "}", "[", "]"]
    
def flatten(t):
    return [item for sublist in t for item in sublist]

def get_ccg_SRL_tokens(input_sentence):

    sentences = sent_tokenize(input_sentence)

    headers = {'Content-type': 'application/json'}
    
    tokens = list()
    sentences_end = list()

    last_end = 0

    for s in sentences:
        SRL_response = requests.post('http://leguin.seas.upenn.edu:4039/annotate/', 
            json={"sentence": s, "task": "tokenize"}, headers=headers)
        if SRL_response.status_code != 200:
            return None, None
        SRL_result = json.loads(SRL_response.text)
        tmp_tokens = [w[0] for w in SRL_result['tokens']]
        last_end = last_end + len(tmp_tokens)
        sentences_end.append(last_end)
        tokens.append(tmp_tokens)
        
    
    tokens = flatten(tokens)
    # print("\ntokens: ", tokens)
    
    SRL_tokens = tokens
    SRL_sentences = {'generator': 'srl_pipeline', 'score': 1.0, 'sentenceEndPositions': sentences_end}
    return SRL_tokens, SRL_sentences, sentences


def Get_CogComp_SRL_results2(input_sentence):
    start_time = time()
    sentences = sent_tokenize(input_sentence)
    
    tokens = list()
    sentences_end = list()
    last_end = 0
    for s in sentences:
        tmp_tokens = word_tokenize(s)
        last_end = last_end + len(tmp_tokens)
        sentences_end.append(last_end)
        tokens.append(tmp_tokens)

    tokens = flatten(tokens)
    SRL_sentences = {'generator': 'srl_pipeline', 'score': 1.0, 'sentenceEndPositions': sentences_end}
    end_time = time()
    print("***Processing Time new:", end_time-start_time)
    return tokens, SRL_sentences, sentences


def Get_CogComp_SRL_results(input_sentence):
    start_time = time()
    # We then work on Celine's SRL system.
    # print('Extracting the events.')
    SRL_tokens = list()
    SRL_sentences = list()
    # SRL_response = requests.get('http://dickens.seas.upenn.edu:4039/annotate', data=input_sentence)
    # start_time_all_srl = time()
    SRL_response = requests.get('http://leguin.seas.upenn.edu:4039/annotate', data=input_sentence)
    # print("Processing Time for SRL backend: ", time() - start_time_all_srl)

    if SRL_response.status_code != 200:
        return None, None
    SRL_result = json.loads(SRL_response.text)
    SRL_tokens = SRL_result['tokens']
    SRL_sentences = SRL_result['sentences']
    end_time = time()
    print("***Processing Time (tokenization): ", end_time-start_time)
    return SRL_tokens, SRL_sentences

    # print('Match tokens.')

def preprocess_input_text(input_text="", multi=False, special_char="remove", char_list=[]):
    # start_time = time()
    input_text = input_text.encode('utf-8').decode("utf-8")
    # special_char_list = ["!", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=", ">", "?", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~"]
    
    if multi:
        if "\n" in input_text:
            input_text = re.sub("\n+", " ", input_text)
        if "\t" in input_text:
            input_text = re.sub("\t+", " ", input_text)

    if special_char=="remove":
        input_text = regex.sub(r'[^\p{Latin}]', u' ', input_text)

    elif special_char=="convert" :
        if "’" in input_text:
            input_text = re.sub("\’", " ' ", input_text)
            # input_text = re.sub("\’", " ", input_text)
        if "‘" in input_text:
            input_text = re.sub("\‘", " ' ", input_text)
            #  input_text = re.sub("\‘", " ", input_text)
        if "“" in input_text:
            input_text = re.sub("“", " \" ", input_text)
            # input_text = re.sub("“", " ", input_text)   
        if "”" in input_text:
            input_text = re.sub("”", " \" ", input_text)
            # input_text = re.sub("”", " ", input_text)
        if "—" in input_text:
            input_text = re.sub("—", " - ", input_text)
        if "–" in input_text:
            input_text = re.sub("–", " - ", input_text)
        if "…" in input_text:
            input_text = re.sub("…", " . ", input_text)
        # if "." in input_text:
        #     input_text = re.sub(".", " . ", input_text)

    if char_list:
        if "'" in input_text and "'" in char_list:
            input_text = re.sub("'", " ' ", input_text)
            # input_text = re.sub("'", " ", input_text)
        if "," in input_text and "," in char_list:
            input_text = re.sub(",", " , ", input_text)
            # input_text = re.sub(",", " ", input_text)
        if ";" in input_text and ";" in char_list:
            input_text = re.sub(";", " ; ", input_text)
            # input_text = re.sub(";", " ", input_text)
        if ":" in input_text and ":" in char_list:
            input_text = re.sub(":", " : ", input_text)
            # input_text = re.sub(":", " ", input_text)
        if "-" in input_text and "-" in char_list:
            input_text = re.sub("-+", " - ", input_text)
        if "?" in input_text and "?" in char_list:
            input_text = re.sub("\?+", " ? ", input_text)
        if "!" in input_text and "!" in char_list:
            input_text = re.sub("!+", " ! ", input_text)

        if "$" in input_text and "$" in char_list:
            input_text = re.sub("\$", " $ ", input_text)
        if "%" in input_text and "%" in char_list:
            input_text = re.sub("%", " % ", input_text)
        if "#" in input_text and "#" in char_list:
            input_text = re.sub("#", " # ", input_text)

        if "_" in input_text and "_" in char_list:
            input_text = re.sub("_", " _ ", input_text)
        if "&" in input_text and "&" in char_list:
            input_text = re.sub("&", " & ", input_text)
        if "~" in input_text and "~"  in char_list:
            input_text = re.sub("~", " ~ ", input_text)
        if "|" in input_text and "|" in char_list:
            input_text = re.sub("|", " | ", input_text)
        if "^" in input_text and "^" in char_list:
            input_text = re.sub("\^", " ^ ", input_text)

        if "+" in input_text and "+" in char_list:
            input_text = re.sub("\+", " + ", input_text)
        if "*" in input_text and "*" in char_list:
            input_text = re.sub("\*", " * ", input_text)
        if "/" in input_text and "/" in char_list:
            input_text = re.sub("/", " / ", input_text)
        if "<" in input_text and "<" in char_list:
            input_text = re.sub("<", " < ", input_text)
        if "=" in input_text and "=" in char_list:
            input_text = re.sub("=", " = ", input_text)
        if ">" in input_text and ">" in char_list:
            input_text = re.sub(">", " > ", input_text)

        if "(" in input_text and "(" in char_list:
            input_text = re.sub("\(", " ( ", input_text)
        if ")" in input_text and ")" in char_list:
            input_text = re.sub("\)", " ) ", input_text)
        if "{" in input_text and "{" in char_list:
            input_text = re.sub("{", " { ", input_text)
        if "}" in input_text and "}" in char_list:
            input_text = re.sub("}", " } ", input_text)
        if "[" in input_text and "[" in char_list:
            input_text = re.sub("\[", " [ ", input_text)
        if "]" in input_text and "]" in char_list:
            input_text = re.sub("\]", " ] ", input_text)
    
    # input_text = regex.sub(r'[^\p{Latin}]', u' ', input_text)
    input_text = re.sub("\s+", " ", input_text).strip()

    # print("input_text[-1] : ", input_text[-1])
    # if not input_text[-1] == '.':
    #     input_text = input_text + "."

    end_time = time()
    # print("***Processing Time (preprocessing): ", time() - start_time)
    return input_text


class MyWebService(object):

    @cherrypy.expose
    def index(self):
        return open('html/index.html', encoding='utf-8')

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def info(self, **params):
        return {"status":"online"}

    @cherrypy.expose
    def halt(self, **params):
        cherrypy.engine.exit()

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def annotate(self):
        start_time = time()
        hasJSON = True
        result = {"status": "false"}
        try:
            # get input JSON
            data = cherrypy.request.json
        except:
            hasJSON = False
            result = {"error": "invalid input"}
        
        if hasJSON:
            # process input
            input_paragraph = data['text']
            input_paragraph = preprocess_input_text(input_paragraph, multi=True, special_char="convert", char_list=special_char_list)
            # input_paragraph = preprocess_input_text(input_paragraph, multi=True, special_char="convert", char_list=["'", ","])
            print("input_paragraph: ", input_paragraph)
            ###
            # headers = {'Content-type': 'application/json'}

#             input_paragraph = re.sub(r'[\n]', ' ', input_paragraph)
            
            # NER_response = requests.post('http://dickens.seas.upenn.edu:4022/ner/',
            #                              json={"task": "ner", "text": "Hello world."}, headers=headers)
            # if NER_response.status_code != 200:
            #     return {'error': 'The NER service is down.'}
            
            # SRL_response = requests.get('http://leguin.seas.upenn.edu:4039/annotate', data=input_paragraph)
            # SRL_response = requests.post('http://leguin.seas.upenn.edu:4039/annotate',
            #                              json={'sentence': "Hello world."})

            # if SRL_response.status_code != 200:
                # return {'error': 'The SRL service is down.'}
            
            # SRL_tokens, SRL_sentences = Get_CogComp_SRL_results(input_paragraph)
            SRL_tokens, SRL_sentences, sentences2 = get_ccg_SRL_tokens(input_paragraph)
            # print("SRL_tokens: ", SRL_tokens)

            # if (not SRL_tokens) or (not SRL_sentences):
            #     return {'error': 'The SRL service is down.'}

            # print(SRL_sentences['sentenceEndPositions'])
            
            
            sentences = list()
            sentences_by_char = list()

            # sentences = sentences2
            # sentences_by_char = SRL_tokens
            
            for i, tmp_s_end_token in enumerate(SRL_sentences['sentenceEndPositions']):
                if i == 0:
                    sentences.append(' '.join(SRL_tokens[:tmp_s_end_token]))
                    sentences_by_char.append(SRL_tokens[:tmp_s_end_token])
                else:
                    sentences.append(' '.join(SRL_tokens[SRL_sentences['sentenceEndPositions'][i-1]:tmp_s_end_token]))
                    sentences_by_char.append(SRL_tokens[SRL_sentences['sentenceEndPositions'][i-1]:tmp_s_end_token])
            if SRL_sentences['sentenceEndPositions'][-1] < len(SRL_tokens):
                sentences.append(' '.join(SRL_tokens[SRL_sentences['sentenceEndPositions'][-1]:]))
                sentences_by_char.append(SRL_tokens[SRL_sentences['sentenceEndPositions'][-1]:])
            
            # sentences = input_paragraph.split('\n')
            
            print('Number of sentences:', len(sentences))
            # print('Number of sentences:', len(SRL_sentences['sentenceEndPositions']))
            
            previous_char = 0
            tmp_view_data = dict()
            tmp_view_data['viewType'] = 'edu.illinois.cs.cogcomp.core.datastructures.textannotation.PredicateArgumentView'
            tmp_view_data['viewName'] = 'event_extraction'
            tmp_view_data['generator'] = 'cogcomp_kairos_event_ie_v1.0'
            tmp_view_data['score'] = 1.0
            tmp_view_data['constituents'] = list()
            tmp_view_data['relations'] = list()
            all_tokens = list()
            sentence_positions = list()
            previous_char = 0
            verb_srl = []
            for s_id, tmp_s in enumerate(sentences):
                # print(s_id, " : ", tmp_s)
                # start_time_event = time()
                extracted_events, tmp_tokens, SRL_sentences2, verb_srl_tmp = extractor.extract(tmp_s, include_all_verbs=True, include_all_nouns=False, demo_version=True)
                # print("Processing Time for 1 sentence event extraction: ", time() - start_time_event)
                verb_srl.append(verb_srl_tmp)
                print(extracted_events)
                # if len(extracted_events) > 0:
                #     tmp_tokens = extracted_events[0]['tokens']
                # else:
                #     tmp_tokens = tmp_s.split(' ')
                all_tokens += tmp_tokens
                sentence_positions.append(len(all_tokens))
                for tmp_event in extracted_events:
                    trigger_start_token_id = tmp_event['trigger']['position'][0] 
                    trigger_end_token_id = tmp_event['trigger']['position'][1]

                    trigger_consituent_position = len(tmp_view_data['constituents'])
                    tmp_view_data['constituents'].append(
                        {'label': tmp_event['trigger']['type'], 'score': 1.0, 'start': (trigger_start_token_id + previous_char),
                         'end': (trigger_end_token_id + previous_char), 'properties': {
                            'SenseNumber': '01', 'sentence_id': s_id,
                                                                     'predicate': tmp_tokens[
                                                                                  trigger_start_token_id:trigger_end_token_id]}})
                    for tmp_argument in tmp_event['arguments']:
                        argument_start_token_id = tmp_argument['position'][0] + previous_char
                        argument_end_token_id = tmp_argument['position'][1] + previous_char
                        tmp_view_data['relations'].append(
                            {'relationName': tmp_argument['role'], 'srcConstituent': trigger_consituent_position,
                             'targetConstituent': len(tmp_view_data['constituents'])})
                        tmp_view_data['constituents'].append(
                            {'label': tmp_argument['role'], 'score': 1.0, 'start': argument_start_token_id,
                             'end': argument_end_token_id, 'entity_type': tmp_argument['entity_type']})
                previous_char += len(sentences_by_char[s_id])

            event_ie_view = dict()
            event_ie_view['viewName'] = 'Event_extraction'
            event_ie_view['viewData'] = [tmp_view_data]

            token_view = dict()
            token_view['viewName'] = 'TOKENS'
            tmp_token_view_data = dict()
            tmp_token_view_data['viewType'] = 'edu.illinois.cs.cogcomp.core.datastructures.textannotation.TokenLabelView'
            tmp_token_view_data['viewName'] = 'TOKENS'
            tmp_token_view_data['generator'] = 'Cogcomp-SRL'
            tmp_token_view_data['score'] = 1.0
            tmp_token_view_data['constituents'] = list()
            for i, tmp_token in enumerate(all_tokens):
                tmp_token_view_data['constituents'].append({'label': tmp_token, 'score': 1.0, 'start': i, 'end': i+1})
            token_view['viewData'] = tmp_token_view_data

            result = dict()
            result['corpusId'] = ''
            result['id'] = ''
            result['text'] = input_paragraph
            result['tokens'] = all_tokens
            result['sentences'] = SRL_sentences
            result['views'] = [token_view, event_ie_view]

            # result['tokens'] = all_tokens
            # result['sentences'] = {'generator': 'srl_pipeline', 'score': 1.0, 'sentenceEndPositions': SRL_sentences['sentenceEndPositions']}
        # return resulting JSON
        end_time = time()
        print("***Processing Time for Event Extraction: ", end_time - start_time)
        print("***Average Time for Event Extraction   : ", (end_time - start_time)/len(SRL_sentences['sentenceEndPositions']))
        # print(result)
        # print("\n")
        if "task" in data and data["task"]=="include_verb_srl":
            return {"result": result, "verb_srl_temporal": verb_srl}
        return result

    @cherrypy.expose
    @cherrypy.tools.json_out()
    @cherrypy.tools.json_in()
    def annotate_multi(self):
        start_time = time()
        hasJSON = True
        result = {"status": "false"}
        try:
            # get input JSON
            data = cherrypy.request.json
        except:
            hasJSON = False
            result = {"error": "invalid input"}
        
        if hasJSON:
            # process input
            input_paragraph = data['text']
            input_paragraph = preprocess_input_text(input_paragraph, multi=True, special_char="convert", char_list=special_char_list)
            # input_paragraph = preprocess_input_text(input_paragraph, multi=True, special_char="convert", char_list=["'", ","])
            print("input_paragraph: ", input_paragraph)
            ###
            # headers = {'Content-type': 'application/json'}

#             input_paragraph = re.sub(r'[\n]', ' ', input_paragraph)
            
            # NER_response = requests.post('http://dickens.seas.upenn.edu:4022/ner/',
            #                              json={"task": "ner", "text": "Hello world."}, headers=headers)
            # if NER_response.status_code != 200:
            #     return {'error': 'The NER service is down.'}
            
            # SRL_response = requests.get('http://leguin.seas.upenn.edu:4039/annotate', data=input_paragraph)
            # SRL_response = requests.post('http://leguin.seas.upenn.edu:4039/annotate',
            #                              json={'sentence': "Hello world."})

            # if SRL_response.status_code != 200:
                # return {'error': 'The SRL service is down.'}
            
            # SRL_tokens, SRL_sentences = Get_CogComp_SRL_results(input_paragraph)
            # SRL_tokens, SRL_sentences, sentences2 = get_ccg_SRL_tokens(input_paragraph)
            # print("SRL_tokens: ", SRL_tokens)

            # if (not SRL_tokens) or (not SRL_sentences):
            #     return {'error': 'The SRL service is down.'}

            # print(SRL_sentences['sentenceEndPositions'])
            
            
            # sentences = list()
            # sentences_by_char = list()

            # sentences = sentences2
            # sentences_by_char = SRL_tokens
            
            # for i, tmp_s_end_token in enumerate(SRL_sentences['sentenceEndPositions']):
            #     if i == 0:
            #         sentences.append(' '.join(SRL_tokens[:tmp_s_end_token]))
            #         sentences_by_char.append(SRL_tokens[:tmp_s_end_token])
            #     else:
            #         sentences.append(' '.join(SRL_tokens[SRL_sentences['sentenceEndPositions'][i-1]:tmp_s_end_token]))
            #         sentences_by_char.append(SRL_tokens[SRL_sentences['sentenceEndPositions'][i-1]:tmp_s_end_token])
            # if SRL_sentences['sentenceEndPositions'][-1] < len(SRL_tokens):
            #     sentences.append(' '.join(SRL_tokens[SRL_sentences['sentenceEndPositions'][-1]:]))
            #     sentences_by_char.append(SRL_tokens[SRL_sentences['sentenceEndPositions'][-1]:])
            
            # sentences = input_paragraph.split('\n')
            
            # print('Number of sentences:', len(sentences))
            # print('Number of sentences:', len(SRL_sentences['sentenceEndPositions']))
            
            previous_char = 0
            tmp_view_data = dict()
            tmp_view_data['viewType'] = 'edu.illinois.cs.cogcomp.core.datastructures.textannotation.PredicateArgumentView'
            tmp_view_data['viewName'] = 'event_extraction'
            tmp_view_data['generator'] = 'cogcomp_kairos_event_ie_v1.0'
            tmp_view_data['score'] = 1.0
            tmp_view_data['constituents'] = list()
            tmp_view_data['relations'] = list()
            all_tokens = list()
            sentence_positions = list()

            # for s_id, tmp_s in enumerate(sentences):
            # print(s_id, " : ", tmp_s)
            # start_time_event = time()
            extracted_events, tmp_tokens, SRL_sentences, verb_srl = extractor.extract(input_paragraph, include_all_verbs=False, include_all_nouns=Fase, demo_version=True)
            # print("Processing Time for 1 sentence event extraction: ", time() - start_time_event)
            
            print(extracted_events)
            # if len(extracted_events) > 0:
            #     tmp_tokens = extracted_events[0]['tokens']
            # else:
            #     tmp_tokens = tmp_s.split(' ')
            all_tokens += tmp_tokens
            # sentence_positions.append(len(all_tokens))
            for tmp_event in extracted_events:

                trigger_start_token_id = tmp_event['trigger']['position'][0] 
                trigger_end_token_id = tmp_event['trigger']['position'][1]
                # print(SRL_sentences['sentenceEndPositions'])
                s_id = -1
                for i in range(len(SRL_sentences['sentenceEndPositions'])):
                    if trigger_end_token_id < SRL_sentences['sentenceEndPositions'][i]:
                        s_id = i-1
                        break
                if s_id==-1:
                    s_id = len(SRL_sentences['sentenceEndPositions']) - 1

                trigger_consituent_position = len(tmp_view_data['constituents'])
                tmp_view_data['constituents'].append(
                    {'label': tmp_event['trigger']['type'], 'score': 1.0, 'start': (trigger_start_token_id),
                        'end': (trigger_end_token_id), 'properties': {
                        'SenseNumber': '01', 'sentence_id': s_id,
                                                                    'predicate': tmp_tokens[
                                                                                trigger_start_token_id:trigger_end_token_id]}})
                for tmp_argument in tmp_event['arguments']:
                    argument_start_token_id = tmp_argument['position'][0]
                    argument_end_token_id = tmp_argument['position'][1] 
                    tmp_view_data['relations'].append(
                        {'relationName': tmp_argument['role'], 'srcConstituent': trigger_consituent_position,
                            'targetConstituent': len(tmp_view_data['constituents'])})
                    tmp_view_data['constituents'].append(
                        {'label': tmp_argument['role'], 'score': 1.0, 'start': argument_start_token_id,
                            'end': argument_end_token_id, 'entity_type': tmp_argument['entity_type']})
            # previous_char += len(sentences_by_char[s_id])

            event_ie_view = dict()
            event_ie_view['viewName'] = 'Event_extraction'
            event_ie_view['viewData'] = [tmp_view_data]

            token_view = dict()
            token_view['viewName'] = 'TOKENS'
            tmp_token_view_data = dict()
            tmp_token_view_data['viewType'] = 'edu.illinois.cs.cogcomp.core.datastructures.textannotation.TokenLabelView'
            tmp_token_view_data['viewName'] = 'TOKENS'
            tmp_token_view_data['generator'] = 'Cogcomp-SRL'
            tmp_token_view_data['score'] = 1.0
            tmp_token_view_data['constituents'] = list()
            for i, tmp_token in enumerate(all_tokens):
                tmp_token_view_data['constituents'].append({'label': tmp_token, 'score': 1.0, 'start': i, 'end': i+1})
            token_view['viewData'] = tmp_token_view_data

            result = dict()
            result['corpusId'] = ''
            result['id'] = ''
            result['text'] = input_paragraph
            result['tokens'] = all_tokens
            result['sentences'] = SRL_sentences
            result['views'] = [token_view, event_ie_view]

            # result['tokens'] = all_tokens
            # result['sentences'] = {'generator': 'srl_pipeline', 'score': 1.0, 'sentenceEndPositions': SRL_sentences['sentenceEndPositions']}
        # return resulting JSON
        end_time = time()
        print("***Processing Time for Event Extraction: ", end_time - start_time)
        print("***Average Time for Event Extraction   : ", (end_time - start_time)/len(SRL_sentences['sentenceEndPositions']))
        # print(result)
        # print("\n")
        return result

if __name__ == '__main__':
    print("")
    # INITIALIZE YOUR MODEL HERE
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default='1', type=str, required=False,
                        help="choose which gpu to use")
    parser.add_argument("--representation_source", default='nyt', type=str, required=False,
                        help="choose which gpu to use")
    parser.add_argument("--model", default='bert-large', type=str, required=False,
                        help="choose which gpu to use")
    parser.add_argument("--pooling_method", default='final', type=str, required=False,
                        help="choose which gpu to use")
    parser.add_argument("--weight", default=100, type=float, required=False,
                        help="weight assigned to triggers")
    parser.add_argument("--argument_matching", default='exact', type=str, required=False,
                        help="weight assigned to triggers")
    parser.add_argument("--eval_model", default='joint', type=str, required=False,
                        help="weight assigned to triggers")

    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('current device:', device)

    extractor = CogcompKairosEventExtractorTest(device, 'mbert')
    # IN ORDER TO KEEP IT IN MEMORY
    print("Starting rest service...")
    cherrypy_cors.install()
    config = {
        'global': {
            'server.socket_host': 'leguin.seas.upenn.edu',
            'server.socket_port': 4023,
            'cors.expose.on': True
        },
        '/': {
            'tools.sessions.on': True,
            'cors.expose.on': True,
            'tools.staticdir.root': os.path.abspath(os.getcwd())

        },
        '/static': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': './html'
        },
        '/html': {
            'tools.staticdir.on': True,
            'tools.staticdir.dir': './html',
            'tools.staticdir.index': 'index.html',
            'tools.gzip.on': True
        }
    }
    cherrypy.config.update(config)
    # cherrypy.config.update({'server.socket_port': 4036})
    cherrypy.quickstart(MyWebService(), '/', config)

# import cherrypy
#
#
# class demoExample:
#     @cherrypy.expose
#     def index(self):
#
#         return "Hello World!!!"
#
#
#
# cherrypy.quickstart(demoExample())
