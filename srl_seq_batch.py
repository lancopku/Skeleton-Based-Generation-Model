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
from nltk.tokenize import sent_tokenize

from nltk import tokenize

FLAGS = tf.app.flags.FLAGS
class Srl_Example(object):


  def __init__(self, text, srl, vocab, hps, mode="None"):
      start_decoding = vocab.word2id(data.START_DECODING)
      stop_decoding = vocab.word2id(data.STOP_DECODING)


      self.hps = hps

      srl_sen_words = tokenize.word_tokenize(srl.strip())
      #shuffle(srl_sen_words)
      if mode == "train" and len(srl_sen_words) > 5:
        srl_sen_words = srl_sen_words[:np.random.randint(5, len(srl_sen_words))]
      if len(srl_sen_words) > hps.srl_max_enc_seq_len:
          srl_sen_words = srl_sen_words[:hps.srl_max_enc_seq_len]

      self.enc_input = [vocab.word2id(w) for w in
                        srl_sen_words]  # list of word ids; OOVs are represented by the id for UNK token

      self.enc_len = len(self.enc_input)


      article_sen = text
      article_sen_words = tokenize.word_tokenize(article_sen.strip())
      if len(article_sen_words) > hps.srl_max_dec_seq_len:
          article_sen_words = article_sen_words[:hps.srl_max_dec_seq_len]


      abs_ids = [vocab.word2id(w) for w in
                 article_sen_words]  # list of word ids; OOVs are represented by the id for UNK token

      # Get the decoder input sequence and target sequence
      self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, hps.srl_max_dec_seq_len,
                                                               start_decoding,
                                                               stop_decoding)  # max_sen_num,max_len, start_doc_id, end_doc_id,start_id, stop_id
      self.dec_len = len(self.dec_input)
      #self.dec_sen_len = [len(sentence) for sentence in self.target]

      self.orig_input = srl
      self.orig_output = text


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

  def pad_decoder_inp_targ(self, max_sen_len, pad_doc_id):
        """Pad decoder input and target sequences with pad_id up to max_len."""




        while len(self.dec_input) < max_sen_len:
            self.dec_input.append(pad_doc_id)



        while len(self.target) < max_sen_len:
            self.target.append(pad_doc_id)


  def pad_encoder_inp_targ(self, max_sen_len, pad_doc_id):
      """Pad decoder input and target sequences with pad_id up to max_len."""


      while len(self.enc_input) < max_sen_len:
          self.enc_input.append(pad_doc_id)







class Srl_Batch(object):
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
          ex.pad_decoder_inp_targ(hps.srl_max_dec_seq_len, self.pad_id)
          ex.pad_encoder_inp_targ(hps.srl_max_enc_seq_len, self.pad_id)
      #pad_encoder_inp_targ(self, max_sen_len, max_sen_num, pad_doc_id):

      # Initialize the numpy arrays.
      # Note: our decoder inputs and targets must be the same length for each batch (second dimension = max_dec_steps) because we do not use a dynamic_rnn for decoding. However I believe this is possible, or will soon be possible, with Tensorflow 1.0, in which case it may be best to upgrade to that.

      self.enc_batch = np.zeros((hps.batch_size, hps.srl_max_enc_seq_len), dtype=np.int32)
      self.enc_lens = np.ones((hps.batch_size), dtype=np.int32)
      #self.dec_lens = np.zeros((hps.batch_size), dtype=np.int32)
      self.dec_batch = np.zeros((hps.batch_size, hps.srl_max_dec_seq_len), dtype=np.int32)
      self.target_batch = np.zeros((hps.batch_size, hps.srl_max_dec_seq_len), dtype=np.int32)
      self.dec_padding_mask = np.zeros((hps.batch_size, hps.srl_max_dec_seq_len),
                                       dtype=np.float32)
      #self.labels = np.zeros((hps.batch_size, hps.max_enc_sen_num, hps.max_enc_seq_len), dtype=np.int32)
      #self.dec_sen_lens = np.zeros((hps.batch_size, hps.srl_max_dec_sen_num), dtype=np.int32)
      self.dec_lens = np.zeros((hps.batch_size), dtype=np.int32)
      self.orig_outputs = []
      self.orig_inputs = []

      for i, ex in enumerate(example_list):
          #self.new_review_text = []
          #self.labels[i]=np.array([[ex.label for k in range(hps.max_enc_seq_len) ] for j in range(hps.max_enc_sen_num)])
          self.orig_outputs.append(ex.orig_output)
          self.orig_inputs.append(ex.orig_input)

          self.dec_lens[i] = ex.dec_len
          self.enc_lens[i] = ex.enc_len
          self.dec_batch[i, :] = np.array(ex.dec_input)
          self.enc_batch[i, :] = np.array(ex.enc_input)
          self.target_batch[i] = np.array(ex.target)


      self.target_batch = np.reshape(self.target_batch,
                                     [hps.batch_size, hps.srl_max_dec_seq_len])


      for i in range(len(self.target_batch)):
          for k in range(len(self.target_batch[i])):
              if int(self.target_batch[i][k]) != self.pad_id:
                  self.dec_padding_mask[i][k] = 1


      self.dec_batch = np.reshape(self.dec_batch, [hps.batch_size, hps.srl_max_dec_seq_len])
      self.dec_lens = np.reshape(self.dec_lens, [hps.batch_size])

      self.enc_batch = np.reshape(self.enc_batch, [hps.batch_size, hps.srl_max_enc_seq_len])
      self.enc_lens = np.reshape(self.enc_lens, [hps.batch_size])
      #self.labels = np.reshape(self.labels, [hps.batch_size * hps.max_enc_sen_num, hps.max_enc_seq_len])





class Srl_GenBatcher(object):
    def __init__(self, vocab, hps):
        self._vocab = vocab
        self._hps = hps

        self.train_queue = self.fill_example_queue("data/0/train.txt", "train")
        self.valid_queue = self.fill_example_queue("data/0/valid.txt", "valid")
        self.test_queue = self.fill_example_queue("data/0/test.txt", "test")

        # self.valid_transfer_queue_negetive = self.fill_example_queue(
        #    "valid/*", mode="valid", target_score=0)


        # self.test_queue = self.fill_example_queue("/home/xujingjing/code/review_summary/dataset/review_generation_dataset/test/*")
        self.train_batch = self.create_batch(mode="train")
        self.valid_batch = self.create_batch(mode="validation", shuffleis=False)
        self.test_batch = self.create_batch(mode="test", shuffleis=False)
        # train_batch = self.create_bach(mode="train")

    def create_batch(self, mode="train", shuffleis=True):
        all_batch = []

        if mode == "train":
            num_batches = int(len(self.train_queue) / self._hps.batch_size)


        elif mode == 'validation':
            num_batches = int(len(self.valid_queue) / self._hps.batch_size)

        elif mode == 'test':
            num_batches = int(len(self.test_queue) / self._hps.batch_size)


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

            all_batch.append(Srl_Batch(batch, self._hps, self._vocab))

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

    def fill_example_queue(self, data_path, mode="None"):

        new_queue = []

        reader = codecs.open(data_path, 'r', 'utf-8')
        j = 0
        while True:
            

            string_ = reader.readline()
            if not string_: break
            dict_example = json.loads(string_)
            srl = dict_example["skeleton"]
            srl = srl.split("|||")

            text = dict_example["text"]
            text = sent_tokenize(text)
            for i,skeleton in enumerate(srl):
                
                example = Srl_Example(text[i], srl[i], self._vocab, self._hps,mode)
                new_queue.append(example)




        return new_queue



