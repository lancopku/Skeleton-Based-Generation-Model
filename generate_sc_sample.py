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


class Generated_sc_sample(object):
    def __init__(self, model, vocab, sess):
        self._model = model
        self._vocab = vocab
        self._sess = sess

    def generator_max_example(self, target_batches, positive_dir, negetive_dir,calbleu=False):

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
        hypothesis = []
        reference = []


        while step < len(target_batches):

            batch = batches[step]
            step += 1

            decode_result = self._model.max_generator(self._sess, batch)

            for i in range(FLAGS.batch_size):

                original_output = batch.orig_outputs[i]
                original_input = batch.orig_input[i]


                output_ids = [int(t) for t in decode_result['generated'][i]][0:]
                #decoded_words = data.outputids2words(output_ids, self._vocab, None)
                # Remove the [STOP] token from decoded_words, if necessary
                try:
                    fst_stop_idx = output_ids.index(4)  # index of the (first) [STOP] symbol
                    output_ids = output_ids[:fst_stop_idx]
                except ValueError:
                    output_ids = output_ids

                decoded_output = [original_input[k] for k in range(len(output_ids)) if output_ids[k] == 1]  # single string

                decoded_output = decoded_output
                if calbleu:
                    hypothesis.append(decoded_output)
                    reference.append([original_output])
                self.write_negtive_temp_to_json(positive_dir, negetive_dir, " ".join(original_input), " ".join(decoded_output))

        if calbleu:
            BLEUscore = nltk.translate.bleu_score.corpus_bleu(reference, hypothesis)
            tf.logging.info("BLEU: "+ str(BLEUscore))

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


    def generator_max_example_test(self, target_batches, negetive_dir):


        self.temp_negetive_dir = negetive_dir


        counter = 0
        batches = target_batches
        step = 0
        write_negetive_file = codecs.open(negetive_dir, "w", "utf-8")
        write_negetive_file.close()



        while step < len(target_batches):

            batch = batches[step]
            step += 1

            decode_result = self._model.max_generator(self._sess, batch)

            for i in range(FLAGS.batch_size):


                original_input = batch.orig_input[i]


                output_ids = [int(t) for t in decode_result['generated'][i]][0:]
                #decoded_words = data.outputids2words(output_ids, self._vocab, None)
                # Remove the [STOP] token from decoded_words, if necessary
                try:
                    fst_stop_idx = output_ids.index(4)  # index of the (first) [STOP] symbol
                    output_ids = output_ids[:fst_stop_idx]
                except ValueError:
                    output_ids = output_ids

                decoded_output = [original_input[k] for k in range(len(output_ids)) if output_ids[k] == 1]  # single string

                decoded_output = decoded_output

                self.write_negtive_temp_to_json_negetive(negetive_dir, " ".join(decoded_output))


    def write_negtive_temp_to_json_negetive(self, negetive_dir,  negetive):


        write_negetive_file = codecs.open(negetive_dir, "a", "utf-8")


        write_negetive_file.write(negetive + "\n")
        write_negetive_file.close()
