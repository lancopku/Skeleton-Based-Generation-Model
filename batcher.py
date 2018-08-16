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
from threading import Thread
import time
import numpy as np
import tensorflow as tf
import data
from nltk.tokenize import sent_tokenize
import glob
import codecs
import json
FLAGS = tf.app.flags.FLAGS
from nltk import tokenize
class Example(object):
  """Class representing a train/val/test example for text summarization."""

  def __init__(self, text,srl, target_sentence, vocab, hps, all_text):
      """Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

          Args:
            article: source text; a string. each token is separated by a single space.
            abstract_sentences: list of strings, one per abstract sentence. In each sentence, each token is separated by a single space.
            vocab: Vocabulary object
            hps: hyperparameters
          """
      self.hps = hps

      # Get ids of special tokens
      start_decoding = vocab.word2id(data.START_DECODING)
      stop_decoding = vocab.word2id(data.STOP_DECODING)
      #stop_doc = vocab.word2id(data.STOP_DECODING_DOCUMENT)

      review_sentence = text
      text_words = []
      for i in range(len(review_sentence)):
          if i >= hps.max_enc_num:
              text_words = text_words[:hps.max_enc_num]
              break
          each_review = review_sentence[i]
          sen_words = tokenize.word_tokenize(each_review.strip())
          if len(sen_words) > hps.max_enc_steps:
              sen_words = sen_words[:hps.max_enc_steps]
          text_words.append(sen_words)

      self.enc_input = [[vocab.word2id(w) for w in sen] for sen in
                        text_words]  # list of word ids; OOVs are represented by the id for UNK token

      self.enc_len = len(self.enc_input)
      self.enc_sen_len = [len(sentence) for sentence in self.enc_input]




      abstract_sen_words = tokenize.word_tokenize(srl.strip())
      if len(abstract_sen_words) > hps.max_dec_steps:
          abstract_sen_words = abstract_sen_words[:hps.max_dec_steps]


      # abstract_words = abstract.split() # list of strings
      abs_ids = [vocab.word2id(w) for w in abstract_sen_words]  # list of word ids; OOVs are represented by the id for UNK token

      # Get the decoder input sequence and target sequence
      self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, hps.max_dec_steps,
                                                               start_decoding,
                                                               stop_decoding)  # max_sen_num,max_len, start_doc_id, end_doc_id,start_id, stop_id
      self.dec_len = len(self.dec_input)
      #self.dec_sen_len = [len(sentence) for sentence in self.target]
      self.original_review_output = srl
      self.original_target_sentence = target_sentence
      self.original_review_input = " ".join(review_sentence)
      self.all_text = " ".join(all_text)



  def get_dec_inp_targ_seqs(self, sequence,max_len, start_id, stop_id):

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

  def pad_decoder_inp_targ(self, max_len, pad_doc_id):
      """Pad decoder input and target sequences with pad_id up to max_len."""

      while len(self.dec_input) < max_len:
          self.dec_input.append(pad_doc_id)


      while len(self.target) < max_len:
          self.target.append(pad_doc_id)



  def pad_encoder_input(self, max_sen_num, max_sen_len, pad_doc_id):
    """Pad the encoder input sequence with pad_id up to max_len."""


    while len(self.enc_sen_len) < max_sen_num:
        self.enc_sen_len.append(1)

    for i in range(len(self.enc_input)):
        while len(self.enc_input[i]) < max_sen_len:
            self.enc_input[i].append(pad_doc_id)

    while len(self.enc_input) < max_sen_num:
        self.enc_input.append([pad_doc_id for i in range(max_sen_len)])




class Batch(object):
  """Class representing a minibatch of train/val/test examples for text summarization."""

  def __init__(self, example_list, hps, vocab):
      """Turns the example_list into a Batch object.

          Args:
             example_list: List of Example objects
             hps: hyperparameters
             vocab: Vocabulary object
          """
      self.pad_id = vocab.word2id(data.PAD_TOKEN)  # id of the PAD token used to pad sequences

      self.init_encoder_seq(example_list, hps)  # initialize the input to the encoder
      self.init_decoder_seq(example_list, hps)  # initialize the input and targets for the decoder
      self.store_orig_strings(example_list)  # store the original strings




  def init_encoder_seq(self, example_list, hps):
      # print ([ex.enc_len for ex in example_list])

      #max_enc_seq_len = max([ex.enc_len for ex in example_list])

      # Pad the encoder input sequences up to the length of the longest sequence
      for ex in example_list:
          ex.pad_encoder_input(  hps.max_enc_num, hps.max_enc_steps, self.pad_id) #(max_enc_seq_len, self.pad_id)

      # Initialize the numpy arrays
      # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
      self.enc_batch = np.zeros((hps.batch_size, hps.max_enc_num, hps.max_enc_steps), dtype=np.int32)
      self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)
      self.enc_sen_lens = np.zeros((hps.batch_size,hps.max_enc_num), dtype=np.int32)
      # self.enc_padding_mask = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.float32)

      # Fill in the numpy arrays
      for i, ex in enumerate(example_list):
          self.enc_batch[i,:, :] = np.array(ex.enc_input)
          self.enc_lens[i] = ex.enc_len
          for j in range(len(ex.enc_sen_len)):
              self.enc_sen_lens[i][j] = ex.enc_sen_len[j]







  def init_decoder_seq(self, example_list, hps):
      for ex in example_list:
          ex.pad_decoder_inp_targ(hps.max_dec_steps, self.pad_id)

      # Initialize the numpy arrays.
      # Note: our decoder inputs and targets must be the same length for each batch (second dimension = max_dec_steps) because we do not use a dynamic_rnn for decoding. However I believe this is possible, or will soon be possible, with Tensorflow 1.0, in which case it may be best to upgrade to that.
      self.dec_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
      self.target_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
      self.dec_padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps),
                                       dtype=np.float32)
      self.dec_lens = np.zeros((hps.batch_size), dtype=np.int32)

      for i, ex in enumerate(example_list):
          self.dec_lens[i] = ex.dec_len
          self.dec_batch[i, :] = np.array(ex.dec_input)
          self.target_batch[i] = np.array(ex.target)


      self.target_batch = np.reshape(self.target_batch,
                                     [hps.batch_size, hps.max_dec_steps])

      for i in range(len(self.target_batch)):
          for k in range(len(self.target_batch[i])):
              if int(self.target_batch[i][k]) != self.pad_id:
                  self.dec_padding_mask[i][k] = 1
                  # self.dec_padding_mask = np.reshape(self.dec_padding_mask, [hps.batch_size*hps.max_dec_sen_num, hps.max_dec_steps])



  def store_orig_strings(self, example_list):
      """Store the original article and abstract strings in the Batch object"""

      self.original_review_outputs = [ex.original_review_output for ex in example_list]  # list of lists
      self.original_review_inputs = [ex.original_review_input for ex in example_list]  # list of lists
      self.orginal_all_text = [ex.all_text for ex in example_list]
      self.original_target_sentences = [ex.original_target_sentence for ex in example_list]
      #self.original_target_texts = [ex.target_text for ex in example_list]  # list of lists
      #self.all_srl = [ex.all_srl for ex in example_list]  # list of lists
      #self.all_text = [ex.all_text for ex in example_list]  # list of lists





class GenBatcher(object):

    def __init__(self, vocab, hps):
        self._vocab = vocab
        self._hps = hps

        self.train_queue = self.fill_example_queue("data/0/train.txt")
        self.valid_queue = self.fill_example_queue("data/0/valid.txt")
        self.test_queue = self.fill_example_queue("data/0/test.txt")

        self.test_valid_queue = self.fill_example_test_queue("data/0/valid.txt")
        self.test_test_queue = self.fill_example_test_queue("data/0/test.txt")

        self.train_batch = self.create_batch(mode="train")
        self.valid_batch = self.create_batch(mode="validation", shuffleis=False)
        self.test_batch = self.create_batch(mode="test", shuffleis=False)

        self.test_valid_batch = self.create_batch(mode="test-validation", shuffleis=False)
        self.test_test_batch = self.create_batch(mode="test-test", shuffleis=False)


    def create_batch(self, mode="train", shuffleis=True):
        all_batch = []

        if mode == "train":
            num_batches = int(len(self.train_queue) / self._hps.batch_size)
            if shuffleis:
                shuffle(self.train_queue)
        elif mode == 'validation':
            num_batches = int(len(self.valid_queue) / self._hps.batch_size)
        elif mode == "test":
            num_batches = int(len(self.test_queue) / self._hps.batch_size)
        elif mode == "test-validation":
            num_batches = int(len(self.test_valid_queue) / self._hps.batch_size)
        elif mode == "test-test":
            num_batches = int(len(self.test_test_queue) / self._hps.batch_size)

        for i in range(0, num_batches):
            batch = []
            if mode == 'train':
                batch += (self.train_queue[i * self._hps.batch_size:i * self._hps.batch_size + self._hps.batch_size])
            elif mode == 'validation':
                batch += (self.valid_queue[i * self._hps.batch_size:i * self._hps.batch_size + self._hps.batch_size])
            elif mode == 'test':
                batch += (self.test_queue[i * self._hps.batch_size:i * self._hps.batch_size + self._hps.batch_size])
            elif mode == "test-validation":
                batch += (self.test_valid_queue[i * self._hps.batch_size:i * self._hps.batch_size + self._hps.batch_size])
            elif mode == 'test-test':
                batch += (self.test_test_queue[i * self._hps.batch_size:i * self._hps.batch_size + self._hps.batch_size])

            all_batch.append(Batch(batch, self._hps, self._vocab))
        return all_batch


    def get_batches(self, mode="train"):


        if mode == "train":
            shuffle(self.train_batch)
            return self.train_batch
        elif mode == 'validation':
            return self.valid_batch
        elif mode == 'test':
            return self.test_batch

        elif mode == 'test-validation':
            return self.test_valid_batch
        elif mode == 'test-test':
            return self.test_test_batch




    def fill_example_test_queue(self, data_path):

        new_queue =[]

        reader = codecs.open(data_path, 'r', 'utf-8')
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
            sentences = sent_tokenize(text)[:-1]
            target_sentences = sent_tokenize(text)

            example = Example([sentences[0]], srl[1], target_sentences[1], self._vocab, self._hps, sent_tokenize(text)[1:])
            new_queue.append(example)

            # example = Example(sent_tokenize(text), "EOD", self._vocab, self._hps)
            # new_queue.append(example)






        return new_queue



    def fill_example_queue(self, data_path):

        new_queue =[]
        j = 0
        reader = codecs.open(data_path, 'r', 'utf-8')
        while True:
            

            string_ = reader.readline()
            if not string_: break
            dict_example = json.loads(string_)

            srl = dict_example["skeleton"]
            srl = srl.split("|||")
            text = dict_example["text"]
            if (len(sent_tokenize(text)) < 2):
                continue
            sentences = sent_tokenize(text)[:-1]
            target_sentences = sent_tokenize(text)
            for i, sen in enumerate(sentences):
            
                if len(srl[i+1].split())<5:
                    continue

                example = Example([sentences[k] for k in range(i+1)], srl[i+1], target_sentences[i+1], self._vocab, self._hps, sent_tokenize(text)[1:])
                new_queue.append(example)

            example = Example(sent_tokenize(text), "EOD", "EOD", self._vocab, self._hps, sent_tokenize(text)[1:])
            new_queue.append(example)






        return new_queue



