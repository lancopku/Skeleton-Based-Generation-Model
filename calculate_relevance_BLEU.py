import codecs
import json
import os
import re
import math
import nltk
import sys
import threading
from nltk.tokenize import sent_tokenize
from nltk.translate.bleu_score import SmoothingFunction
from nltk.corpus import stopwords



def read_gold_true(read_file):

    new_queue = []
    new_out_queue = []

    reader = codecs.open(read_file, 'r', 'utf-8')
    j = 0
    while True:

        string_ = reader.readline()
        if not string_: break
        dict_example = json.loads(string_)

        srl = dict_example["skeleton"]
        srl = srl.split("|||")
        text = dict_example["text"]
        if (len(sent_tokenize(text)) < 2):
            continue
        sentences = sent_tokenize(text)[0]
        new_queue.append(sentences)
        new_out_queue.append(" ".join(sent_tokenize(text)[1:]))
    print ("length of gold test input: " + str(len(new_queue)))
    sys.stdout.flush()


    return new_queue, new_out_queue

total_bleu = 0
words = stopwords.words('english')
def read_generator(read_file, gold_inputs, gold_outputs):
    content = read_file.readlines()
    print (len(content))
    bleu_h = []
    bleu_r = []
    
    bleu_h_io = []
    bleu_r_io = []
    global words
    
    for i in range(len(content)):
        gold_inp = gold_inputs[i]
        gold_out = gold_outputs[i]
    
        output = json.loads(content[i].strip())
        output = output["example"]
        output = sent_tokenize(output)
        if len(output)>2:
            output = output[:2]
        gold_out = sent_tokenize(gold_out)
        if len(gold_out)>2:
            gold_out = gold_out[:2]
        gold_out = " ".join(gold_out)
        output = " ".join(output)
        #output = sent_tokenize(text)[0:1]
        
        
       
        bleu_h.append([[w for w in output.split() if(w not in words)]])
        bleu_r.append([w for w in gold_out.split() if(w not in words)])
        
        prediction = gold_inp + " " + output
        pre_sentences = sent_tokenize(prediction)
        for j in range(len(pre_sentences)-1):
            bleu_h_io.append([[w for w in pre_sentences[j].split() if(w not in words)]])
            bleu_r_io.append([w for w in pre_sentences[j+1].split() if(w not in words)])
    smoother = SmoothingFunction()    
    BLEUscore_1 = nltk.translate.bleu_score.corpus_bleu(bleu_h_io,bleu_r_io,smoothing_function=smoother.method1)
    BLEUscore_2 = nltk.translate.bleu_score.corpus_bleu(bleu_h,bleu_r,  weights=(0.25,0.25,0.25,0.25),  smoothing_function=smoother.method1)
    print(BLEUscore_2)
    BLEUscore_2 = nltk.translate.bleu_score.corpus_bleu(bleu_h,bleu_r,  weights=(0.33,0.33,0.33,0),  smoothing_function=smoother.method1)
    print(BLEUscore_2)
    BLEUscore_2 = nltk.translate.bleu_score.corpus_bleu(bleu_h,bleu_r,  weights=(0.5,0.5,0,0),  smoothing_function=smoother.method1)
    print(BLEUscore_2)
    BLEUscore_2 = nltk.translate.bleu_score.corpus_bleu(bleu_h,bleu_r,  weights=(1,0,0,0),  smoothing_function=smoother.method1)
    print(BLEUscore_2)
   
    sys.stdout.flush()
    print(BLEUscore_2)
    sys.stdout.flush()
    





def read_train_pair(read_file_path):
    read_file = codecs.open(read_file_path, "r", "utf-8")
    content = read_file.readlines()
    sentence_pair_input = []
    sentence_pair_output = []
    sentence_pair_merge = []
    for i in range(len(content)):
        sentences = sent_tokenize(content[i].strip())
        if len(sentences) < 2:
            continue
        for k in range(len(sentences)-1):
            sentence_pair_input.append(sentences[k].split())
            sentence_pair_output.append(sentences[k+1].split())
            sentence_pair_merge.append((sentences[k]+" "+sentences[k+1]).split())
    return sentence_pair_input, sentence_pair_output, sentence_pair_merge




gold_input, gold_output = read_gold_true("data/0/test.txt")

path = "max_generated_final/test/"
for i in range(31,0,-1):  
    print("the iterations: " +str(i))
    read_file = codecs.open(path+ str(i) + "_negative/result.txt", "r", "utf-8")
    read_generator(read_file, gold_input, gold_output)