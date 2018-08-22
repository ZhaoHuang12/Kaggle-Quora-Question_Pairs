import tensorflow as tf 
import numpy as np 
import sys

from data_helper import load_data, pad_seq_id, batch_iter
from models.CNNBase import CNNBase
from models.CNNSiamese import CNNSiamese
from models.LSTMSiamese_Cosinedistance import SiameseLSTM

import os
import time
import datetime
import pickle

#if __name__ == '__main__':
#data loading params and model hyperparams, training params
tf.flags.DEFINE_float("val_sample_percentage", 0.9, "percentage of the data used for training")
tf.flags.DEFINE_string("training_data_file", "inputs/train.csv", "data source for training data")
tf.flags.DEFINE_string("filter_sizes", '3,4,5', "comma-separated filter sizes")
tf.flags.DEFINE_integer("num_filters", 128, "number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "dropout keep probability")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda")
tf.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.flags.DEFINE_integer("batch_size", 128, "batch_size")
tf.flags.DEFINE_integer("num_epochs", 20, "number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 500, "evalute model on val set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 500, "save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 10, "number of checkpoints to store")

#if we use LSTMSiamese_CosineDistance model
tf.flags.DEFINE_integer("rnn_size", 200, "hidden size of rnn cell")
tf.flags.DEFINE_integer("num_layers", 2, "number of stacked rnn layers")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)


print("loading training data...")
processed_question1, processed_question2, label, word2id, word_embeddings = load_data(FLAGS.training_data_file)
#convert word2 to ints in vocabulary
s1, s2 = pad_seq_id(processed_question1, processed_question2, word2id)

#save for later use
fw = open('processed_inputs.txt','wb')
pickle.dump(s1, fw)
pickle.dump(s2, fw)
pickle.dump(label, fw)
pickle.dump(word_embeddings, fw)
pickle.dump(word2id, fw)
fw.close()


#load processed inputs
f = open("processed_inputs.txt", "rb")
s1 = pickle.load(f)
s2 = pickle.load(f)
label = pickle.load(f)
word_embeddings = pickle.load(f)
word2id = pickle.load(f)
f.close()
 


print ("shuffle the training data...")
np.random.seed(10)
shuffled_indices = np.random.permutation(np.arange(len(s1)))
s1 = [s1[i] for i in shuffled_indices]
s2 = [s2[i] for i in shuffled_indices]
label = [label[i] for i in shuffled_indices]

print ("splitng the data into training set and validation set...")
training_end = int(len(label)*FLAGS.val_sample_percentage)
s1_train = s1[:training_end]
s1_val = s1[training_end:]
s2_train = s2[:training_end]
s2_val = s2[training_end:]
label_train = label[:training_end]
label_val = label[training_end:]
print ("Training/Val split: {:d}/{:d}".format(len(s1_train), len(s1_val)))
sequence_length = len(s1_val[0])
num_classes = len(label[0])
print ("start training....")


  
with tf.Session() as sess:
  # Define Training procedure
  model = CNNSiamese(filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))), num_filters = FLAGS.num_filters, sequence_length = sequence_length, 
                 num_classes = num_classes, pretrained_embeddings = word_embeddings, l2_reg_lamda = FLAGS.l2_reg_lambda)
  
  # model = CNNBase(filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))), num_filters = FLAGS.num_filters, sequence_length = sequence_length, 
  #                num_classes = num_classes, pretrained_embeddings = word_embeddings, l2_reg_lamda = FLAGS.l2_reg_lambda)

  # rnn = SiameseLSTM(hidden_units = FLAGS.rnn_size, num_layers = FLAGS.num_layers, sequence_length  = sequence_length,
  #                   pretrained_embeddings = word_embeddings)

  #initialize all variables
  init_op = tf.global_variables_initializer()
  sess.run(init_op)

  #keep track of graidients and sparsity
  grad_summaries = []
  for g, v in model.grads_and_vars:
    if g is not None:
      grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name),g)
      sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
      grad_summaries.append(grad_hist_summary)
      grad_summaries.append(sparsity_summary)
  grad_summaries_merged = tf.summary.merge(grad_summaries)

  #Output directory for models and summaries
  timestamp = str(int(time.time()))
  out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
  print("Writing to {}\n".format(out_dir))

  #summary for loss and accuracy
  loss_summary = tf.summary.scalar("loss", model.loss)
  accuracy_summary = tf.summary.scalar("accuracy", model.accuracy)

  #train summary
  train_summary_op = tf.summary.merge([loss_summary, accuracy_summary, grad_summaries_merged]) 
  train_summary_dir = os.path.join(out_dir, "summaries", "train") 
  train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph) 

  #validation summary
  val_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
  val_summary_dir = os.path.join(out_dir, "summaries", "val")
  val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

  #checkpoint directory
  checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints")) 
  checkpoint_prefix = os.path.join(checkpoint_dir, "model")
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  saver = tf.train.Saver(tf.global_variables(), max_to_keep = FLAGS.num_checkpoints)

  data_train = list(zip(s1_train, s2_train, label_train))
  for epoch in range(FLAGS.num_epochs):
    batches = batch_iter(data_train, FLAGS.batch_size, shuffle = True)
    for batch in batches:
      s1_batch, s2_batch, label_batch = zip(*batch)
      feed_train = {
                    model.input1: s1_batch,
                    model.input2: s2_batch,
                    model.y:      label_batch,
                    model.learning_rate: FLAGS.learning_rate,
                    model.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }

      _, step, loss, accuracy, summaries = sess.run(
                  [model.train_op, model.global_step, model.loss, model.accuracy, train_summary_op],
                  feed_dict = feed_train)

      time_str = datetime.datetime.now().isoformat()
      print("{}: \tepoch {}, \tstep {}, \tloss {:g}, \tacc {:g}".format(time_str, epoch, step, loss, accuracy))
      train_summary_writer.add_summary(summaries, step)


      if step % FLAGS.evaluate_every == 0:
        print("\nEvaluation:")
        loss_val = []
        accuracy_val = []
        data_val = list(zip(s1_val, s2_val, label_val))
        batches_val = batch_iter(data_val, FLAGS.batch_size, shuffle = False)
        for batch_val in batches_val:
          s1_batch_val, s2_batch_val, label_batch_val = zip(*batch_val)
          feed_val = {
                    model.input1: s1_batch_val,
                    model.input2: s2_batch_val,
                    model.y: label_batch_val,
                    model.dropout_keep_prob: 1.0
                  }
          step_val, summaries_val, loss1, accuracy1 = sess.run([model.global_step, val_summary_op, model.loss, model.accuracy],
                                                    feed_dict = feed_val)
          val_summary_writer.add_summary(summaries_val, step_val)
          loss_val.append(loss1)
          accuracy_val.append(accuracy1)

        time_str_val = datetime.datetime.now().isoformat()
        print("{}: step_val {}, loss_val {:g}, acc_val {:g}".format(time_str_val, step_val, np.mean(loss_val), np.mean(accuracy_val)))
          
              
      if step % FLAGS.checkpoint_every == 0:
        path = saver.save(sess, checkpoint_prefix, global_step=step)
        print("Saved model checkpoint to {}\n".format(path))




