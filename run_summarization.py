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

"""This is the top-level file to train, evaluate or test your summarization model"""

import sys
from random import shuffle
import time
import codecs
import data
import os
import math
import tensorflow as tf
import numpy as np
from collections import namedtuple
from data import Vocab
from nltk.tokenize import sent_tokenize
from batcher import Example
from batcher import Batch
from srl_seq_batch import Srl_Batch
from srl_seq_batch import Srl_GenBatcher
from srl_seq_batch import Srl_Example
from srl_seq_model import Srl_Generator
from batcher import GenBatcher
from model import Generator
from sc_model import Sc_Generator
from sc_batch import Sc_GenBatcher
from sc_batch import Sc_Batch

import json
from generated_sample import  Generated_sample
from generator_whole import Generated_whole_sample
from generated_srl_sample import Generated_srl_sample
from generate_sc_sample import Generated_sc_sample
import util
import re
import nltk
from result_evaluate import Evaluate
from tensorflow.python import debug as tf_debug

FLAGS = tf.app.flags.FLAGS

# Where to find data
#tf.app.flags.DEFINE_string('data_path', '', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_string('dataset', 'story_dataset', '')

# Where to save output
tf.app.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')


tf.app.flags.DEFINE_integer('gpuid', 0, 'for gradient clipping')

tf.app.flags.DEFINE_integer('max_enc_steps', 40, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_enc_num', 6, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_sen_num', 6, 'max timesteps of decoder (max source text tokens)')   # for generator
tf.app.flags.DEFINE_integer('max_dec_steps', 15, 'max timesteps of decoder (max source text tokens)')   # for generator


tf.app.flags.DEFINE_integer('sc_max_enc_seq_len', 40, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('sc_max_dec_seq_len', 40, 'max timesteps of decoder (max source text tokens)')   # for generator



tf.app.flags.DEFINE_integer('srl_max_enc_sen_num', 6, 'max timesteps of decoder (max source text tokens)')   # for srl_seq
tf.app.flags.DEFINE_integer('srl_max_enc_seq_len', 15, 'max timesteps of decoder (max source text tokens)')   # for generator
tf.app.flags.DEFINE_integer('srl_max_dec_sen_num', 6, 'max timesteps of decoder (max source text tokens)')   # for srl_seq
tf.app.flags.DEFINE_integer('srl_max_dec_seq_len', 40, 'max timesteps of decoder (max source text tokens)')   # for generator

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 128, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 50, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 10, 'minibatch size')

tf.app.flags.DEFINE_integer('vocab_size', 20000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.6, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad') # for discriminator and generator
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization') # for discriminator and generator
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else') # for discriminator and generator
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping') # for discriminator and generator


'''the generator model is saved at FLAGS.log_root + "train-generator"
   give up sv, use sess
'''
def setup_training_generator(model):
  """Does setup before starting training (run_training)"""
  train_dir = os.path.join(FLAGS.log_root, "train-generator")
  if not os.path.exists(train_dir): os.makedirs(train_dir)

  model.build_graph() # build the graph

  saver = tf.train.Saver(max_to_keep=20)  # we use this to load checkpoints for decoding
  sess = tf.Session(config=util.get_config())
  init = tf.global_variables_initializer()

  sess.run(init)
  #tf.get_variable_scope().reuse_variables()

  # Load an initial checkpoint to use for decoding
  #util.load_ckpt(saver, sess, ckpt_dir="train-generator")


  return sess, saver,train_dir

def setup_training_srl_generator(model):
  """Does setup before starting training (run_training)"""
  train_dir = os.path.join(FLAGS.log_root, "train-srl-generator")
  if not os.path.exists(train_dir): os.makedirs(train_dir)

  model.build_graph() # build the graph

  saver = tf.train.Saver(max_to_keep=20)  # we use this to load checkpoints for decoding
  sess = tf.Session(config=util.get_config())
  init = tf.global_variables_initializer()
  sess.run(init)
  #tf.get_variable_scope().reuse_variables()

  # Load an initial checkpoint to use for decoding
  #util.load_ckpt(saver, sess, ckpt_dir="train-srl-generator")

  return sess, saver,train_dir


def setup_training_sc_generator(model):
  """Does setup before starting training (run_training)"""
  train_dir = os.path.join(FLAGS.log_root, "train-sc-generator")
  if not os.path.exists(train_dir): os.makedirs(train_dir)

  model.build_graph() # build the graph

  saver = tf.train.Saver(max_to_keep=20)  # we use this to load checkpoints for decoding
  sess = tf.Session(config=util.get_config())
  init = tf.global_variables_initializer()
  sess.run(init)
  #util.load_ckpt(saver, sess, ckpt_dir="train-sc-generator")

  #tf.get_variable_scope().reuse_variables()

  return sess, saver, train_dir

  # Load an initial checkpoint to use for decoding


def print_batch(batch):
    tf.logging.info("enc_batch")
    tf.logging.info(list(batch.enc_batch))
    #tf.logging.info("enc_pos_batch")
    #tf.logging.info(list(batch.enc_pos_batch))
    #tf.logging.info("enc_dep_batch")
    #tf.logging.info(list(batch.enc_dep_batch))
    tf.logging.info("enc_lens")
    tf.logging.info(list(batch.enc_lens))
    # tf.logging.info("enc_sen_lens")
    # tf.logging.info(list(batch.enc_sen_lens))


    tf.logging.info('dec_batch')
    tf.logging.info(list(batch.dec_batch))

    tf.logging.info('target_batch')
    tf.logging.info(list(batch.target_batch))

    tf.logging.info('dec_padding_mask')
    tf.logging.info(list(batch.dec_padding_mask))
    
    
    tf.logging.info('dec_lens')
    tf.logging.info(list(batch.dec_lens))
    #tf.logging.info(batch.original_reviews)





def run_pre_train_generator(model, batcher, max_run_epoch, sess, saver, train_dir, generated):
    tf.logging.info("starting run_pre_train_generator")
    epoch = 0
    while epoch < max_run_epoch:
        batches = batcher.get_batches(mode='train')
        step = 0
        t0 = time.time()
        loss_window = 0.0
        while step < len(batches):
            current_batch = batches[step]
            #print_batch(current_batch)
            step += 1
            results = model.run_pre_train_step(sess, current_batch)
            loss = results['loss']
            loss_window += loss

            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")

            train_step = results['global_step']  # we need this to update our running average loss
            if train_step % 1000 == 0:
                t1 = time.time()
                tf.logging.info('seconds for %d training generator step: %.3f ', train_step, (t1 - t0) / 1000)
                t0 = time.time()
                tf.logging.info('loss: %f', loss_window / 1000)  # print the loss to screen
                loss_window = 0.0
            if train_step % 10000 == 0:
                saver.save(sess, train_dir + "/model", global_step=train_step)
                if not os.path.exists("to_srl_max_generated/"): os.mkdir("to_srl_max_generated/")
                if not os.path.exists("to_srl_max_generated/validation/"): os.mkdir("to_srl_max_generated/validation/")
                if not os.path.exists("to_srl_max_generated/test/"): os.mkdir("to_srl_max_generated/test/")
                generated.generator_max_example(batcher.get_batches("validation"), "to_srl_max_generated/valid/"+str(int(train_step / 10000))+"_positive", "to_srl_max_generated/valid/"+str(int(train_step / 10000))+"_negative")
                generated.generator_max_example(batcher.get_batches("test"),
                                                  "to_srl_max_generated/test/" + str(int(train_step / 10000)) + "_positive",
                                                  "to_srl_max_generated/test/" + str(int(train_step / 10000)) + "_negative")


        epoch += 1
        tf.logging.info("finished %d epoches", epoch)

def run_pre_train_srl_generator(model, batcher, srl_batcher, max_run_epoch, sess, saver, train_dir, generated, whole_generated):
    tf.logging.info("starting run_pre_train_generator")
    epoch = 0
    while epoch < max_run_epoch:
        batches = srl_batcher.get_batches(mode='train')
        step = 0
        t0 = time.time()
        loss_window = 0.0
        while step < len(batches):
            current_batch = batches[step]
            #print_batch(current_batch)
            step += 1
            results = model.run_pre_train_step(sess, current_batch)
            loss = results['loss']
            loss_window += loss

            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")

            train_step = results['global_step']  # we need this to update our running average loss
            if train_step % 1000 == 0:
                t1 = time.time()
                tf.logging.info('seconds for %d training generator step: %.3f ', train_step, (t1 - t0) / 1000)
                t0 = time.time()
                tf.logging.info('loss: %f', loss_window / 1000)  # print the loss to screen
                loss_window = 0.0
            if train_step % 30000 == 0:
                saver.save(sess, train_dir + "/model", global_step=train_step)
                
                if not os.path.exists("to_seq_max_generated/"): os.mkdir("to_seq_max_generated/")
                if not os.path.exists("max_generated_final/"): os.mkdir("max_generated_final/")
                if not os.path.exists("to_seq_max_generated/test/"): os.mkdir("to_seq_max_generated/test/")
                if not os.path.exists("max_generated_final/test/"): os.mkdir("max_generated_final/test/")
                if not os.path.exists("to_seq_max_generated/valid/"): os.mkdir("to_seq_max_generated/valid/")
                if not os.path.exists("max_generated_final/valid/"): os.mkdir("max_generated_final/valid/")
                generated.generator_max_example(srl_batcher.get_batches("validation"), "to_seq_max_generated/valid/"+str(int(train_step / 30000))+"_positive", "to_seq_max_generated/valid/"+str(int(train_step / 30000))+"_negative")
                generated.generator_max_example(srl_batcher.get_batches("test"),
                                                  "to_seq_max_generated/test/" + str(int(train_step / 30000)) + "_positive",
                                                  "to_seq_max_generated/test/" + str(int(train_step / 30000)) + "_negative")

                whole_generated.generator_max_example(batcher.get_batches("test-validation"),
                                                "max_generated_final/valid/" + str(int(train_step / 30000)) + "_positive",
                                                "max_generated_final/valid/" + str(int(train_step / 30000)) + "_negative")
                whole_generated.generator_max_example(batcher.get_batches("test-test"),
                                                "max_generated_final/test/" + str(int(train_step / 30000)) + "_positive",
                                                "max_generated_final/test/" + str(int(train_step / 30000)) + "_negative")



        epoch += 1
        tf.logging.info("finished %d epoches", epoch)

def run_pre_train_sc_generator(model, batcher, max_run_epoch, sess, saver, train_dir, generated):
    tf.logging.info("starting run_pre_train_generator")
    epoch = 0
    while epoch < max_run_epoch:
        batches = batcher.get_batches(mode='train')
        step = 0
        t0 = time.time()
        loss_window = 0.0
        while step < len(batches):
            current_batch = batches[step]
            #print_batch(current_batch)
            step += 1
            results = model.run_pre_train_step(sess, current_batch)
            loss = results['loss']
            loss_window += loss

            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")

            train_step = results['global_step']  # we need this to update our running average loss
            if train_step % 100 == 0:
                t1 = time.time()
                tf.logging.info('seconds for %d training generator step: %.3f ', train_step, (t1 - t0) / 100)
                t0 = time.time()
                tf.logging.info('loss: %f', loss_window / 100)  # print the loss to screen
                loss_window = 0.0
            if train_step % 1000 == 0:
                saver.save(sess, train_dir + "/model", global_step=train_step)
                if not os.path.exists("to_sc_max_generated/"): os.mkdir("to_sc_max_generated/")
                if not os.path.exists("to_sc_max_generated/valid/"): os.mkdir("to_sc_max_generated/valid/")
                if not os.path.exists("to_sc_max_generated/test/"): os.mkdir("to_sc_max_generated/test/")
                generated.generator_max_example(batcher.get_batches("validation"),
                                                "to_sc_max_generated/valid/" + str(
                                                    int(train_step / 1000)) + "_positive",
                                                "to_sc_max_generated/valid/" + str(
                                                    int(train_step / 1000)) + "_negative", calbleu = True)
                generated.generator_max_example(batcher.get_batches("test"),
                                                "to_sc_max_generated/test/" + str(
                                                    int(train_step / 1000)) + "_positive",
                                                "to_sc_max_generated/test/" + str(
                                                    int(train_step / 1000)) + "_negative", calbleu = True)

        epoch += 1
        tf.logging.info("finished %d epoches", epoch)
# def batch_to_batch(batch, batcher, srl_batcher):
#
#     srl_example_list = []
#
#     for i in range(FLAGS.batch_size):
#
#         new_srl_example = Srl_Example(batch.original_review_outputs[i], batch.original_target_texts, srl_batcher._vocab, srl_batcher._hps)
#         srl_example_list.append(new_srl_example)
#
#     return Srl_Batch(srl_example_list, srl_batcher._hps, srl_batcher._vocab)

# def output_to_batch(current_batch, result, batcher, srl_batcher):
#     example_list= []
#     srl_example_list = []
#
#     for i in range(FLAGS.batch_size):
#         decoded_words_all = []
#         encode_words = current_batch.original_review_inputs[i]
#
#         for j in range(FLAGS.max_dec_sen_num):
#
#             output_ids = [int(t) for t in result['generated'][i][j]][1:]
#             decoded_words = data.outputids2words(output_ids, batcher._vocab, None)
#             # Remove the [STOP] token from decoded_words, if necessary
#             try:
#                 fst_stop_idx = decoded_words.index(data.STOP_DECODING)  # index of the (first) [STOP] symbol
#                 decoded_words = decoded_words[:fst_stop_idx]
#             except ValueError:
#                 decoded_words = decoded_words
#             if len(decoded_words) < 2:
#                 continue
#
#
#             decoded_output = ' '.join(decoded_words).strip()  # single string
#             decoded_words_all.append(decoded_output)
#
#
#         decoded_words_all = ' ||| '.join(decoded_words_all).strip()
#         try:
#             fst_stop_idx = decoded_words_all.index(
#                 data.STOP_DECODING_DOCUMENT)  # index of the (first) [STOP] symbol
#             decoded_words_all = decoded_words_all[:fst_stop_idx]
#         except ValueError:
#             decoded_words_all = decoded_words_all
#         decoded_words_all = decoded_words_all.replace("[UNK] ", "")
#         decoded_words_all = decoded_words_all.replace("[UNK]", "")
#         decoded_words_all, _ = re.subn(r"(! ){2,}", "", decoded_words_all)
#         decoded_words_all, _ = re.subn(r"(\. ){2,}", "", decoded_words_all)
#
#         if decoded_words_all.strip() == "":
#             '''tf.logging.info("decode")
#             tf.logging.info(current_batch.original_reviews[i])
#             tf.logging.info("encode")
#             tf.logging.info(encode_words)'''
#             new_dis_example = Srl_Example(current_batch.original_review_outputs[i], current_batch.original_target_texts[i],  srl_batcher._vocab, srl_batcher._hps)
#             new_example = Example(current_batch.all_text[i], current_batch.original_target_texts[i],  batcher._vocab, batcher._hps)
#
#         else:
#             '''tf.logging.info("decode")
#             tf.logging.info(decoded_words_all)
#             tf.logging.info("encode")
#             tf.logging.info(encode_words)'''
#             new_dis_example = Srl_Example(current_batch.original_review_outputs[i], decoded_words_all.split("|||"), srl_batcher._vocab, srl_batcher._hps)
#             new_example = Example(current_batch.all_text[i], decoded_words_all.split("|||"), batcher._vocab, batcher._hps)
#         example_list.append(new_example)
#         srl_example_list.append(new_dis_example)
#
#    return Batch(example_list, batcher._hps, batcher._vocab), Srl_Batch(srl_example_list, srl_batcher._vocab, srl_batcher._hps)

def merge(f_read_path, f_read_skeleton, f_write_path):
    f_read_text = codecs.open(f_read_path, "r", "utf-8")
    f_read_result = codecs.open(f_read_skeleton, "r", "utf-8")

    f_write_result_parallel = codecs.open(f_write_path, "w", "utf-8")

    def read_true_sen(skeletons):
        content = f_read_text.readlines()
        i = 0
        for story in content:
            # parse_seq_new = ""
            skeleton_new = ""

            story_sens = sent_tokenize(story)
            for sen in story_sens:
                if i >= len(skeletons):
                    f_write_result_parallel.close()
                    return
                skeleton_new += " ||| " + skeletons[i]
                i += 1

            jsObj = json.dumps({"text": story.strip(), "skeleton": skeleton_new[5:].strip()})
            f_write_result_parallel.write(jsObj + "\n")
        f_write_result_parallel.close()

    def read_skeleton_seq():
        content = f_read_result.readlines()
        process_result = []
        for sentence in content:
            sent = sentence.strip()
            if sent =="":
                sent = "was"
            process_result.append(sent)
        return process_result

    skeleton = read_skeleton_seq()
    read_true_sen(skeleton)
def main(unused_argv):
    if len(unused_argv) != 1:  # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    tf.logging.set_verbosity(tf.logging.INFO)  # choose what level of logging you want
    tf.logging.info('Starting running in %s mode...', (FLAGS.mode))

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
    if not os.path.exists(FLAGS.log_root):
        if FLAGS.mode == "train":
            os.makedirs(FLAGS.log_root)
        else:
            raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

    vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)  # create a vocabulary

    # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
    hparam_list = ['vocab_size', 'dataset', 'mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag',
                   'trunc_norm_init_std', 'max_grad_norm', 'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_sen_num', 'max_enc_num',
                   'max_dec_steps', 'max_enc_steps']
    hps_dict = {}
    for key, val in FLAGS.__flags.items():  # for each flag
        if key in hparam_list:  # if it's in the list
            hps_dict[key] = val  # add it to the dict
    hps_generator = namedtuple("HParams", hps_dict.keys())(**hps_dict)



    hparam_list = ['lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
                   'hidden_dim', 'emb_dim', 'batch_size', 'srl_max_dec_seq_len', 'srl_max_dec_sen_num', 'srl_max_enc_seq_len', 'srl_max_enc_sen_num']
    hps_dict = {}
    for key, val in FLAGS.__flags.items():  # for each flag
        if key in hparam_list:  # if it's in the list
            hps_dict[key] = val  # add it to the dict
    hps_srl_generator = namedtuple("HParams", hps_dict.keys())(**hps_dict)

    hparam_list = ['lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
                   'hidden_dim', 'emb_dim', 'batch_size', 'sc_max_dec_seq_len',
                   'sc_max_enc_seq_len']
    hps_dict = {}
    for key, val in FLAGS.__flags.items():  # for each flag
        if key in hparam_list:  # if it's in the list
            hps_dict[key] = val  # add it to the dict
    hps_sc_generator = namedtuple("HParams", hps_dict.keys())(**hps_dict)






    # Create a batcher object that will create minibatches of data

    sc_batcher = Sc_GenBatcher(vocab, hps_sc_generator)






    tf.set_random_seed(111)  # a seed value for randomness

    if hps_generator.mode == 'train':

        print("Start pre-training......")
        sc_model = Sc_Generator(hps_sc_generator, vocab)

        sess_sc, saver_sc, train_dir_sc = setup_training_sc_generator(sc_model)
        sc_generated = Generated_sc_sample(sc_model, vocab, sess_sc)
        print("Start pre-training generator......")
        run_pre_train_sc_generator(sc_model, sc_batcher, 40, sess_sc, saver_sc, train_dir_sc, sc_generated)

        if not os.path.exists("data/" + str(0) + "/"): os.mkdir("data/" + str(0) + "/")
        sc_generated.generator_max_example_test(sc_batcher.get_batches("pre-train"),
        
                                         "data/" + str(
                                             0) + "/train_skeleton.txt")
        
        sc_generated.generator_max_example_test(sc_batcher.get_batches("pre-valid"),
        
                                            "data/" + str(
                                                0) + "/valid_skeleton.txt")

        sc_generated.generator_max_example_test(sc_batcher.get_batches("pre-test"),
        
                                            "data/" + str(
                                                0) + "/test_skeleton.txt")


        merge("data/story/train_process.txt", "data/0/train_skeleton.txt", "data/0/train.txt")
        merge("data/story/validation_process.txt", "data/0/valid_skeleton.txt", "data/0/valid.txt")
        merge("data/story/test_process.txt", "data/0/test_skeleton.txt", "data/0/test.txt")


        #################################################################################################
        batcher = GenBatcher(vocab, hps_generator)
        srl_batcher = Srl_GenBatcher(vocab, hps_srl_generator)
        print("Start pre-training......")
        model = Generator(hps_generator, vocab)

        sess_ge, saver_ge, train_dir_ge = setup_training_generator(model)
        generated = Generated_sample(model, vocab, sess_ge)
        print("Start pre-training generator......")
        run_pre_train_generator(model, batcher, 30, sess_ge, saver_ge, train_dir_ge, generated)
        ##################################################################################################
        srl_generator_model = Srl_Generator(hps_srl_generator, vocab)

        sess_srl_ge, saver_srl_ge, train_dir_srl_ge = setup_training_srl_generator(srl_generator_model)
        util.load_ckpt(saver_ge,sess_ge,ckpt_dir="train-generator")
        util.load_ckpt(saver_sc, sess_sc, ckpt_dir="train-sc-generator")
        srl_generated = Generated_srl_sample(srl_generator_model, vocab, sess_srl_ge)
        whole_generated = Generated_whole_sample(model, srl_generator_model, vocab, sess_ge, sess_srl_ge, batcher, srl_batcher)
        print("Start pre-training srl_generator......")
        run_pre_train_srl_generator(srl_generator_model, batcher,srl_batcher, 20, sess_srl_ge, saver_srl_ge, train_dir_srl_ge, srl_generated, whole_generated)

        loss_window = 0
        t0 = time.time()
        print("begin reinforcement learning:")
        for epoch in range(10):

            loss_window = 0.0

            batcher = GenBatcher(vocab, hps_generator)
            srl_batcher = Srl_GenBatcher(vocab, hps_srl_generator)

            batches = batcher.get_batches(mode='train')
            srl_batches = srl_batcher.get_batches(mode='train')
            sc_batches = sc_batcher.get_batches(mode = 'train')
            len_sc = len(sc_batches)

            for i in range(min(len(batches), len(srl_batches))):
                current_batch = batches[i]
                current_srl_batch = srl_batches[i]
                current_sc_batch = sc_batches[i%(len_sc-1)]

                results = model.run_pre_train_step(sess_ge, current_batch)
                loss_list = results['without_average_loss']

                example_skeleton_list = current_batch.original_review_outputs
                example_text_list = current_batch.original_target_sentences

                new_batch = sc_batcher.get_text_queue(example_skeleton_list, example_text_list, loss_list)
                results_sc = sc_model.run_rl_train_step(sess_sc, new_batch)
                loss = results_sc['loss']
                loss_window += loss

                results_srl = srl_generator_model.run_pre_train_step(sess_srl_ge, current_srl_batch)
                loss_list_srl = results_srl['without_average_loss']

                example_srl_text_list = current_srl_batch.orig_outputs
                example_skeleton_srl_list = current_srl_batch.orig_inputs

                new_batch = sc_batcher.get_text_queue(example_skeleton_srl_list, example_srl_text_list, loss_list_srl)
                results_sc = sc_model.run_rl_train_step(sess_sc, new_batch)
                loss = results_sc['loss']
                loss_window += loss

                results_sc = sc_model.run_rl_train_step(sess_sc, current_sc_batch)
                loss = results_sc['loss']
                loss_window += loss




                train_step = results['global_step']

                if train_step % 100 == 0:
                    t1 = time.time()
                    tf.logging.info('seconds for %d training generator step: %.3f ', train_step, (t1 - t0) / 300)
                    t0 = time.time()
                    tf.logging.info('loss: %f', loss_window / 100)  # print the loss to screen
                    loss_window = 0.0
                
                train_srl_step = results_srl['global_step']
                    
                if train_srl_step % 10000 == 0:
                    saver_sc.save(sess_sc, train_dir_sc + "/model", global_step=results_sc['global_step'])
                    saver_ge.save(sess_ge, train_dir_ge + "/model", global_step=train_step)
                    saver_srl_ge.save(sess_srl_ge, train_dir_srl_ge + "/model", global_step=train_srl_step)
                    
                    
                    srl_generated.generator_max_example(srl_batcher.get_batches("validation"), "to_seq_max_generated/valid/"+str(int(train_srl_step / 30000))+"_positive", "to_seq_max_generated/valid/"+str(int(train_srl_step / 30000))+"_negative")
                    srl_generated.generator_max_example(srl_batcher.get_batches("test"),
                                                      "to_seq_max_generated/test/" + str(int(train_srl_step / 30000)) + "_positive",
                                                      "to_seq_max_generated/test/" + str(int(train_srl_step / 30000)) + "_negative")
    
                    whole_generated.generator_max_example(batcher.get_batches("test-validation"),
                                                    "max_generated_final/valid/" + str(int(train_srl_step / 30000)) + "_positive",
                                                    "max_generated_final/valid/" + str(int(train_srl_step / 30000)) + "_negative")
                    whole_generated.generator_max_example(batcher.get_batches("test-test"),
                                                    "max_generated_final/test/" + str(int(train_srl_step / 30000)) + "_positive",
                                                    "max_generated_final/test/" + str(int(train_srl_step / 30000)) + "_negative")

            sc_generated.generator_max_example_test(sc_batcher.get_batches("pre-train"),

                                                    "data/" + str(
                                                        0) + "/train_skeleton.txt")

            sc_generated.generator_max_example_test(sc_batcher.get_batches("pre-valid"),

                                                    "data/" + str(
                                                        0) + "/valid_skeleton.txt")

            sc_generated.generator_max_example_test(sc_batcher.get_batches("pre-test"),

                                                    "data/" + str(
                                                        0) + "/test_skeleton.txt")

            merge("data/story/train_process.txt", "data/0/train_skeleton.txt", "data/0/train.txt")
            merge("data/story/validation_process.txt", "data/0/valid_skeleton.txt", "data/0/valid.txt")
            merge("data/story/test_process.txt", "data/0/test_skeleton.txt", "data/0/test.txt")
            


    else:
        raise ValueError("The 'mode' flag must be one of train/eval/decode")

if __name__ == '__main__':
  tf.app.run()
