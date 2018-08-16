import os
import json
import time
import codecs
import tensorflow as tf
import data
import shutil
import util
import re
from  result_evaluate import Evaluate
import nltk
from nltk.translate.bleu_score import corpus_bleu
FLAGS = tf.app.flags.FLAGS

class Generated_sample(object):
    def __init__(self, model, vocab, sess):
        self._model = model
        self._vocab = vocab
        self._sess = sess

    # def generator_test_max_example(self, target_batches, positive_dir, negetive_dir):
    #
    #     self.temp_positive_dir = positive_dir
    #     self.temp_negetive_dir = negetive_dir
    #
    #     if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
    #     if not os.path.exists(self.temp_negetive_dir): os.mkdir(self.temp_negetive_dir)
    #     shutil.rmtree(self.temp_negetive_dir)
    #     shutil.rmtree(self.temp_positive_dir)
    #     if not os.path.exists(self.temp_positive_dir): os.mkdir(self.temp_positive_dir)
    #     if not os.path.exists(self.temp_negetive_dir): os.mkdir(self.temp_negetive_dir)
    #     counter = 0
    #     batches = target_batches
    #     step = 0
    #
    #     while step < len(target_batches):
    #
    #         batch = batches[step]
    #         step += 1
    #
    #         decode_result = self._model.max_generator(self._sess, batch)
    #
    #         for i in range(FLAGS.batch_size):
    #
    #             decoded_words_all = []
    #             original_review = batch.original_review_outputs[i]
    #
    #             for j in range(FLAGS.max_dec_sen_num):
    #
    #                 output_ids = [int(t) for t in decode_result['generated'][i][j]][0:]
    #                 decoded_words = data.outputids2words(output_ids, self._vocab, None)
    #                 # Remove the [STOP] token from decoded_words, if necessary
    #                 try:
    #                     fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
    #                     decoded_words = decoded_words[:fst_stop_idx]
    #                 except ValueError:
    #                     decoded_words = decoded_words
    #
    #                 if len(decoded_words) < 2:
    #                     continue
    #
    #                 decoded_output = ' '.join(decoded_words).strip()  # single string
    #                 decoded_words_all.append(decoded_output)
    #             decoded_words_all = ' ||| '.join(decoded_words_all).strip()
    #             try:
    #                 fst_stop_idx = decoded_words_all.index(
    #                     data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
    #                 decoded_words_all = decoded_words_all[:fst_stop_idx]
    #             except ValueError:
    #                 decoded_words_all = decoded_words_all
    #             decoded_words_all = decoded_words_all.replace("[UNK] ", "")
    #             decoded_words_all = decoded_words_all.replace("[UNK]", "")
    #             decoded_words_all, _ = re.subn(r"(! ){2,}", "! ", decoded_words_all)
    #             decoded_words_all, _ = re.subn(r"(\. ){2,}", ". ", decoded_words_all)
    #             self.write_negtive_temp_to_json(positive_dir, negetive_dir, original_review, decoded_words_all)
    #
    #             # eva = Evaluate()
    #             # eva.diversity_evaluate(negetive_dir + "/*")

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

            batch = batches[step]
            step += 1

            decode_result = self._model.max_generator(self._sess, batch)


            for i in range(FLAGS.batch_size):

                decoded_words_all = []
                original_review = batch.original_review_outputs[i]



                output_ids = [int(t) for t in decode_result['generated'][i]][0:]
                decoded_words = data.outputids2words(output_ids, self._vocab, None)
                # Remove the [STOP] token from decoded_words, if necessary
                try:
                    fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
                    decoded_words = decoded_words[:fst_stop_idx]
                except ValueError:
                    decoded_words = decoded_words




                decoded_output = ' '.join(decoded_words).strip()  # single string
                decoded_words_all = decoded_output
                #decoded_words_all = ' ||| '.join(decoded_words_all).strip()
                try:
                    fst_stop_idx = decoded_words_all.index(
                        data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
                    decoded_words_all = decoded_words_all[:fst_stop_idx]
                except ValueError:
                    decoded_words_all = decoded_words_all
                decoded_words_all = decoded_words_all.replace("[UNK] ", "")
                decoded_words_all = decoded_words_all.replace("[UNK]", "")
                decoded_words_all, _ = re.subn(r"(! ){2,}", "! ", decoded_words_all)
                decoded_words_all, _ = re.subn(r"(\. ){2,}", ". ", decoded_words_all)
                self.write_negtive_temp_to_json(positive_dir, negetive_dir, original_review, decoded_words_all)

        #eva = Evaluate()
        #eva.diversity_evaluate(negetive_dir + "/*")


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
