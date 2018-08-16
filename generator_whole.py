import os
import json
import time
import copy
import numpy as np
import codecs
import tensorflow as tf
import data
import shutil
import util
import re
from  result_evaluate import Evaluate
from srl_seq_batch import Srl_Example
from srl_seq_batch import Srl_Batch
from nltk import tokenize
import nltk
from nltk.translate.bleu_score import corpus_bleu

FLAGS = tf.app.flags.FLAGS


class Generated_whole_sample(object):
    def __init__(self, model, srl_model, vocab, sess, sess_srl, batcher, srl_bathcher):
        self._model = model
        self._vocab = vocab
        self._sess = sess
        self._srl_model = srl_model
        self._sess_srl = sess_srl
        self._batcher = batcher
        self._srl_batcher = srl_bathcher

    def output_to_batch(self,current_batch, result):

        srl_example_list = []
        decode_mask = []

        for i in range(FLAGS.batch_size):

            output_ids = [int(t) for t in result['generated'][i]][0:]
            decoded_words = data.outputids2words(output_ids, self._batcher._vocab, None)
            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            decoded_output = ' '.join(decoded_words).strip()  # single string


            try:
                fst_stop_idx = decoded_output.index(
                    data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
                decoded_output = decoded_output[:fst_stop_idx]
            except ValueError:
                decoded_output = decoded_output
            decoded_output = decoded_output.replace("[UNK] ", "")
            decoded_output = decoded_output.replace("[UNK]", "")
            decoded_output, _ = re.subn(r"(! ){2,}", "", decoded_output)
            decoded_output, _ = re.subn(r"(\. ){2,}", "", decoded_output)

            if decoded_output.strip() == "":
                new_dis_example = Srl_Example(current_batch.original_review_outputs[i], "was",
                                              self._srl_batcher._vocab, self._srl_batcher._hps)
                decode_mask.append(0)

            else:
                new_dis_example = Srl_Example(current_batch.original_review_outputs[i], decoded_output,
                                              self._srl_batcher._vocab, self._srl_batcher._hps)
                decode_mask.append(1)


            srl_example_list.append(new_dis_example)

        return Srl_Batch(srl_example_list, self._srl_batcher._hps, self._srl_batcher._vocab), decode_mask



    def seq_output_to_batch(self, decode_result_seq, batch):

        for i in range(FLAGS.batch_size):

            #original_review = batch.original_review_outputs[i]

            output_ids = [int(t) for t in decode_result_seq['generated'][i]][0:]
            decoded_words = data.outputids2words(output_ids, self._vocab, None)

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            decoded_output = ' '.join(decoded_words).strip()  # single string

            try:
                fst_stop_idx = decoded_output.index(
                    data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
                decoded_output = decoded_output[:fst_stop_idx]
            except ValueError:
                decoded_output = decoded_output
                decoded_output = decoded_output.replace("[UNK] ", "")
                decoded_output = decoded_output.replace("[UNK]", "")
                decoded_output, _ = re.subn(r"(! ){2,}", "! ", decoded_output)
                decoded_output, _ = re.subn(r"(\. ){2,}", ". ", decoded_output)

            abstract_sen_words = tokenize.word_tokenize(decoded_output.strip())
            if len(abstract_sen_words) > FLAGS.max_enc_steps:
                abstract_sen_words = abstract_sen_words[:FLAGS.max_enc_steps]

            # abstract_words = abstract.split() # list of strings
            enc_ids = [self._vocab.word2id(w) for w in
                       abstract_sen_words]  # list of word ids; OOVs are represented by the id for UNK token

            batch.enc_lens[i] = batch.enc_lens[i] + 1
            batch.enc_sen_lens[i][batch.enc_lens[i] - 1] = len(enc_ids)

            while len(enc_ids) < FLAGS.max_enc_steps:
                enc_ids.append(self._vocab.word2id(data.PAD_TOKEN))


            batch.enc_batch[i,batch.enc_lens[i]-1,:] = enc_ids


        return batch


            #inputs, targets = get_dec_inp_targ_seqs(sequence, max_len, start_id, stop_id)





    def generator_max_example(self, target_batches, positive_dir, negetive_dir):

        self.temp_positive_dir = positive_dir
        self.temp_negetive_dir = negetive_dir

        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
        if not os.path.exists(self.temp_negetive_dir): os.mkdir(self.temp_negetive_dir)
        shutil.rmtree(self.temp_negetive_dir)
        shutil.rmtree(self.temp_positive_dir)
        if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
        if not os.path.exists(self.temp_negetive_dir): os.mkdir(self.temp_negetive_dir)
        counter = 0
        batches = target_batches
        step = 0

        while step < len(target_batches):

            batch = copy.deepcopy(batches[step])
            step += 1
            decoded_words_all = [[]  for i in range(FLAGS.batch_size)]
            original_reviews = [ " ".join(review) for review in  batch.orginal_all_text]
            #tf.logging.info(batch.enc_lens)

            for k in range(FLAGS.max_dec_sen_num):

                decode_result = self._model.max_generator(self._sess, batch)

                srl_batch, decode_mask = self.output_to_batch(batch, decode_result)
                decode_result_seq = self._srl_model.max_generator(self._sess_srl, srl_batch)
                if k < FLAGS.max_dec_sen_num-1:

                    batch = self.seq_output_to_batch(decode_result_seq, batch)

                for i in range(FLAGS.batch_size):

                    if decode_mask[i] == 0:
                        decoded_words_all[i].append(data.STOP_DECODING_DOCUMENT)
                    else:
                        output_ids = [int(t) for t in decode_result_seq['generated'][i]][0:]
                        decoded_words = data.outputids2words(output_ids, self._vocab, None)
                        # Remove the [STOP] token from decoded_words, if necessary
                        try:
                            fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                            decoded_words = decoded_words[:fst_stop_idx]
                        except ValueError:
                            decoded_words = decoded_words

                        if len(decoded_words) < 1:
                            continue

                        if len(decoded_words_all[i]) > 0:
                            new_set1 = set(decoded_words_all[i][len(decoded_words_all[i]) - 1].split())
                            new_set2 = set(decoded_words)
                            if len(new_set1 & new_set2) > 0.5 * len(new_set2):
                                continue
                        decoded_output = ' '.join(decoded_words).strip()  # single string
                        decoded_words_all[i].append(decoded_output)

            for i in range(FLAGS.batch_size):
                batch_seq = ' '.join([decoded_words_all[i][j] for j in range(len(decoded_words_all[i]))]).strip()
                try:
                    fst_stop_idx = batch_seq.index(
                        data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
                    batch_seq = batch_seq[:fst_stop_idx]
                except ValueError:
                    batch_seq = batch_seq
                batch_seq = batch_seq.replace("[UNK] ", "")
                batch_seq = batch_seq.replace("[UNK]", "")
                batch_seq, _ = re.subn(r"(! ){2,}", "! ", batch_seq)
                batch_seq, _ = re.subn(r"(\. ){2,}", ". ", batch_seq)
                self.write_negtive_temp_to_json(positive_dir, negetive_dir, original_reviews[i], batch_seq)

        eva = Evaluate()
        eva.diversity_evaluate(negetive_dir + "/*")

    def write_negtive_temp_to_json(self, positive_dir, negetive_dir, positive, negetive):
        positive_file = os.path.join(positive_dir, "result.txt")
        negetive_file = os.path.join(negetive_dir, "result.txt")
        write_positive_file = codecs.open(positive_file, "a", "utf-8")
        write_negetive_file = codecs.open(negetive_file, "a", "utf-8")
        dict = {"example": str(positive),
                "label": str(1)
                }
        string_ = json.dumps(dict)
        write_positive_file.write(string_ + "\n")

        dict = {"example": str(negetive),
                "label": str(0)
                }
        string_ = json.dumps(dict)
        write_negetive_file.write(string_ + "\n")
        write_negetive_file.close()
        write_positive_file.close()
