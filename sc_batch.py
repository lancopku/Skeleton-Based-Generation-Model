# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
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
# ==============================================================================

"""This file contains code to process data into batches"""

import queue
from random import shuffle
import codecs
import json
import glob
import numpy as np
import tensorflow as tf
import data
import spacy
from nltk.tokenize import sent_tokenize

from nltk import tokenize

FLAGS = tf.app.flags.FLAGS
class Sc_Example(object):


  def __init__(self, text, label, pos_tagging, compression, dependency,  vocab, hps, reward = 1):
      start_decoding = vocab.word2id(data.START_DECODING)
      stop_decoding = vocab.word2id(data.STOP_DECODING)


      self.hps = hps

      text_sen_words = text
      if len(text_sen_words) > hps.sc_max_enc_seq_len:
          text_sen_words = text_sen_words[:hps.sc_max_enc_seq_len]

      self.enc_input_text = [vocab.word2id(w) for w in
                        text_sen_words]  # list of word ids; OOVs are represented by the id for UNK token

      self.enc_len = len(self.enc_input_text)
      self.orig_input = text

      pos_words = pos_tagging
      if len(pos_words) > hps.sc_max_enc_seq_len:
          pos_words = pos_words[:hps.sc_max_enc_seq_len]

      self.enc_input_pos = [vocab.word2id_add(w) for w in
                             pos_words]  # list of word ids; OOVs are represented by the id for UNK token

      dependency_words = dependency
      if len(dependency_words) > hps.sc_max_enc_seq_len:
          dependency_words = dependency_words[:hps.sc_max_enc_seq_len]

      self.enc_input_dep = [vocab.word2id_add(w) for w in
                            dependency_words]  # list of word ids; OOVs are represented by the id for UNK token



      if len(label) > hps.sc_max_dec_seq_len:
          label = label[:hps.sc_max_dec_seq_len]


      abs_ids = []
      for w in label:
          if isinstance(w, list):
              abs_ids.append(0)
          else:
              abs_ids.append(int(w))

      #abs_ids = [int(w) for w in
      #           label]  # list of word ids; OOVs are represented by the id for UNK token

      # Get the decoder input sequence and target sequence
      self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, hps.sc_max_dec_seq_len,
                                                               3,
                                                               4)  # max_sen_num,max_len, start_doc_id, end_doc_id,start_id, stop_id
      self.dec_len = len(self.dec_input)
      #self.dec_sen_len = [len(sentence) for sentence in self.target]

      self.orig_input = text
      self.orig_output = compression
      self.label_output = label
      self.reward = reward


  def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        """Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).

        Args:
          sequence: List of ids (integers)
          max_len: integer
          start_id: integer
          stop_id: integer

        Returns:
          inp: sequence length <=max_len starting with start_id
          target: sequence same length as input, ending with stop_id only if there was no truncation
        """

        inps = sequence[:]
        targets = sequence[:]


        inps = [start_id] + inps[:]
        if len(inps) > max_len:
            inps = inps[:max_len]

        if len(targets) >= max_len:
            targets = targets[:max_len - 1]  # no end_token
            targets.append(stop_id)  # end token
        else:
            targets = targets + [stop_id]

        return inps, targets

  def pad_decoder_inp_targ(self, max_sen_len):
        """Pad decoder input and target sequences with pad_id up to max_len."""




        while len(self.dec_input) < max_sen_len:
            self.dec_input.append(2)



        while len(self.target) < max_sen_len:
            self.target.append(2)


  def pad_encoder_inp_targ(self, max_sen_len, pad_doc_id):
      """Pad decoder input and target sequences with pad_id up to max_len."""


      while len(self.enc_input_text) < max_sen_len:
          self.enc_input_text.append(pad_doc_id)


      while len(self.enc_input_pos) < max_sen_len:
          self.enc_input_pos.append(pad_doc_id)

      while len(self.enc_input_dep) < max_sen_len:
          self.enc_input_dep.append(pad_doc_id)







class Sc_Batch(object):
  """Class representing a minibatch of train/val/test examples for text summarization."""

  def __init__(self, example_list, hps, vocab):
    """Turns the example_list into a Batch object.

    Args:
       example_list: List of Example objects
       hps: hyperparameters
       vocab: Vocabulary object
    """
    self.pad_id = vocab.word2id(data.PAD_TOKEN) # id of the PAD token used to pad sequences
    self.init_decoder_seq(example_list, hps)  # initialize the input to the encoder

  def init_decoder_seq(self, example_list, hps):

      for ex in example_list:
          ex.pad_decoder_inp_targ(hps.sc_max_dec_seq_len)
          ex.pad_encoder_inp_targ(hps.sc_max_enc_seq_len, self.pad_id)
      #pad_encoder_inp_targ(self, max_sen_len, max_sen_num, pad_doc_id):

      # Initialize the numpy arrays.
      # Note: our decoder inputs and targets must be the same length for each batch (second dimension = max_dec_steps) because we do not use a dynamic_rnn for decoding. However I believe this is possible, or will soon be possible, with Tensorflow 1.0, in which case it may be best to upgrade to that.

      self.enc_text_batch = np.zeros((hps.batch_size, hps.sc_max_enc_seq_len), dtype=np.int32)
      self.enc_pos_batch = np.zeros((hps.batch_size, hps.sc_max_enc_seq_len), dtype=np.int32)
      self.enc_dep_batch = np.zeros((hps.batch_size, hps.sc_max_enc_seq_len), dtype=np.int32)
      self.enc_lens = np.ones((hps.batch_size), dtype=np.int32)
      #self.dec_lens = np.zeros((hps.batch_size), dtype=np.int32)
      self.dec_batch = np.zeros((hps.batch_size, hps.sc_max_dec_seq_len), dtype=np.int32)
      self.target_batch = np.zeros((hps.batch_size, hps.sc_max_dec_seq_len), dtype=np.int32)
      self.dec_padding_mask = np.zeros((hps.batch_size, hps.sc_max_dec_seq_len),
                                       dtype=np.float32)
                                       
      #self.labels = np.zeros((hps.batch_size, hps.max_enc_sen_num, hps.max_enc_seq_len), dtype=np.int32)
      #self.dec_sen_lens = np.zeros((hps.batch_size, hps.srl_max_dec_sen_num), dtype=np.int32)
      self.dec_lens = np.zeros((hps.batch_size), dtype=np.int32)
      self.orig_outputs = []
      self.label_outputs = []
      self.orig_input = []
      self.rewards = np.zeros((hps.batch_size), dtype=np.int32)
      for i, ex in enumerate(example_list):
          #self.new_review_text = []
          #self.labels[i]=np.array([[ex.label for k in range(hps.max_enc_seq_len) ] for j in range(hps.max_enc_sen_num)])
          self.orig_outputs.append(ex.orig_output)
          self.label_outputs.append(ex.label_output)
          self.orig_input.append(ex.orig_input)

          self.dec_lens[i] = ex.dec_len
          self.rewards[i] = ex.reward
          self.dec_batch[i, :] = np.array(ex.dec_input)
          self.enc_text_batch[i, :] = np.array(ex.enc_input_text)
          self.enc_pos_batch[i, :] = np.array(ex.enc_input_pos)
          self.enc_dep_batch[i, :] = np.array(ex.enc_input_dep)
          self.target_batch[i] = np.array(ex.target)
          self.enc_lens[i] = np.array(ex.enc_len)


      self.target_batch = np.reshape(self.target_batch,
                                     [hps.batch_size, hps.sc_max_dec_seq_len])


      for i in range(len(self.target_batch)):
          for k in range(len(self.target_batch[i])):
              if int(self.target_batch[i][k]) != 2:
                  self.dec_padding_mask[i][k] = 1


      self.dec_batch = np.reshape(self.dec_batch, [hps.batch_size, hps.sc_max_dec_seq_len])
      self.dec_lens = np.reshape(self.dec_lens, [hps.batch_size])

      self.enc_text_batch = np.reshape(self.enc_text_batch, [hps.batch_size, hps.sc_max_enc_seq_len])
      self.enc_pos_batch = np.reshape(self.enc_pos_batch, [hps.batch_size, hps.sc_max_enc_seq_len])
      self.enc_dep_batch = np.reshape(self.enc_dep_batch, [hps.batch_size, hps.sc_max_enc_seq_len])
      self.enc_lens = np.reshape(self.enc_lens, [hps.batch_size])
      #self.labels = np.reshape(self.labels, [hps.batch_size * hps.max_enc_sen_num, hps.max_enc_seq_len])





class Sc_GenBatcher(object):
    def __init__(self, vocab, hps):
        self._vocab = vocab
        self._hps = hps
        self.nlp = spacy.load('en')


        self.train_queue = self.fill_example_queue("data/trainfeature02.json")
        self.valid_queue = self.fill_example_queue("data/validfeature02.json")
        self.test_queue = self.fill_example_queue("data/testfeature02.json")

        self.predict_train_queue = self.fill_example_queue("data/story/train_sc.txt")
        self.predict_valid_queue = self.fill_example_queue("data/story/valid_sc.txt")
        self.predict_test_queue = self.fill_example_queue("data/story/test_sc.txt")

        # self.valid_transfer_queue_negetive = self.fill_example_queue(
        #    "valid/*", mode="valid", target_score=0)


        # self.test_queue = self.fill_example_queue("/home/xujingjing/code/review_summary/dataset/review_generation_dataset/test/*")
        self.train_batch = self.create_batch(mode="train")
        self.valid_batch = self.create_batch(mode="validation", shuffleis=False)
        self.test_batch = self.create_batch(mode="test", shuffleis=False)

        self.predict_train_batch = self.create_batch(mode="pre-train", shuffleis=False)
        self.predict_valid_batch = self.create_batch(mode="pre-valid", shuffleis=False)
        self.predict_test_batch = self.create_batch(mode="pre-test", shuffleis=False)

        # train_batch = self.create_bach(mode="train")

    def create_batch(self, mode="train", shuffleis=True):
        all_batch = []

        if mode == "train":
            num_batches = int(len(self.train_queue) / self._hps.batch_size)


        elif mode == 'validation':
            num_batches = int(len(self.valid_queue) / self._hps.batch_size)

        elif mode == 'test':
            num_batches = int(len(self.test_queue) / self._hps.batch_size)

        elif mode == 'pre-train':
            num_batches = int(len(self.predict_train_queue) / self._hps.batch_size)

        elif mode == 'pre-valid':
            num_batches = int(len(self.predict_valid_queue) / self._hps.batch_size)

        elif mode == 'pre-test':
            num_batches = int(len(self.predict_test_queue) / self._hps.batch_size)


        for i in range(0, num_batches):
            batch = []
            if mode == 'train':
                batch += (
                self.train_queue[i * self._hps.batch_size:i * self._hps.batch_size + self._hps.batch_size])
            elif mode == 'validation':
                batch += (
                self.valid_queue[i * self._hps.batch_size:i * self._hps.batch_size + self._hps.batch_size])

            elif mode == 'test':
                batch += (
                self.test_queue[i * self._hps.batch_size:i * self._hps.batch_size + self._hps.batch_size])

            elif mode ==  'pre-train':
                batch += (
                self.predict_train_queue[i * self._hps.batch_size:i * self._hps.batch_size + self._hps.batch_size])

            elif mode ==  'pre-valid':
                batch += (
                self.predict_valid_queue[i * self._hps.batch_size:i * self._hps.batch_size + self._hps.batch_size])

            elif mode ==  'pre-test':
                batch += (
                self.predict_test_queue[i * self._hps.batch_size:i * self._hps.batch_size + self._hps.batch_size])

            all_batch.append(Sc_Batch(batch, self._hps, self._vocab))

        if mode == "train" and shuffleis:
            shuffle(all_batch)

        return all_batch




    def get_batches(self, mode="train"):

        if mode == "train":
            shuffle(self.train_batch)
            return self.train_batch
        elif mode == 'validation':
            return self.valid_batch
        elif mode == 'test':
            return self.test_batch
        elif mode == 'pre-train':
            return self.predict_train_batch
        elif mode == 'pre-valid':
            return self.predict_valid_batch
        elif mode == 'pre-test':
            return self.predict_test_batch



    def get_tokenize(self,inputx, inputy):
            listx = []
            list_pos = []
            list_dep = []
            doc = self.nlp(inputx)
            for token in doc:
                listx += [token.lower_]
                list_pos += [token.pos_]
                list_dep += [token.dep_]
    
            listy = []
            doc = self.nlp(inputy)
            for token in doc:
                listy += [token.lower_]
            return listx, listy, list_pos, list_dep

    def lcs_base(self, input_x, input_y):
        position = []

        if len(input_x) == 0 or len(input_y) == 0:
            return position, [], [], []
        indexx = 0
        for i in range(len(input_y)):
            if indexx > len(input_x) - 1:
                return position
            if input_y[i] not in input_x[indexx:]:
                continue
            while not input_y[i] == input_x[indexx]:
                position.append("0")
                indexx += 1
                if indexx > len(input_x) - 1:

                    return position
            position.append("1")
            indexx += 1

        return position

    def get_text_queue(self, skeletons, texts, rewards):

        new_queue = []


        for i in range(len(texts)):

            sentence = texts[i]
            skeleton = skeletons[i]


            sen_list, com_list, list_pos, list_dep = self.get_tokenize(sentence, skeleton)
            if len(sen_list) > 0:
                position = self.lcs_base(sen_list, com_list)

                example = Sc_Example(sentence, position, list_pos, skeleton, list_dep, self._vocab, self._hps, reward = (1-rewards[i])*(1-rewards[i]))
                new_queue.append(example)
            else:
                new_queue.append(self.train_queue[i])

        batch = Sc_Batch(new_queue[0:], self._hps, self._vocab)





        return batch






    def fill_example_queue(self, data_path):

        new_queue = []

        reader = codecs.open(data_path, 'r', 'utf-8')
        j = 0
        while True:

            string_ = reader.readline()
            if not string_: break
            dict_example = json.loads(string_)


            text = dict_example["text"]
            label = dict_example["label"]

            pos_tagging = dict_example["pos tagging"]
            compression = dict_example["compression"]
            dependency = dict_example["dependency"]

            example = Sc_Example(text, label, pos_tagging, compression, dependency, self._vocab, self._hps)
            new_queue.append(example)




        return new_queue




    def fill_example_queue(self, data_path):

        new_queue = []

        reader = codecs.open(data_path, 'r', 'utf-8')
        j = 0
        while True:
            

            string_ = reader.readline()
            if not string_: break
            dict_example = json.loads(string_)


            text = dict_example["text"]
            label = dict_example["label"]
            pos_tagging = dict_example["pos tagging"]
            compression = dict_example["compression"]
            dependency = dict_example["dependency"]
            #print(text)
            #print (label)

            example = Sc_Example(text, label, pos_tagging, compression, dependency, self._vocab, self._hps)
            new_queue.append(example)




        return new_queue



