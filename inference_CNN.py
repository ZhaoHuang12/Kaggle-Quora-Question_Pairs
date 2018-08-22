import tensorflow as tf 
import numpy as np 
import sys
from data_helper import batch_iter, clean_str
from model.CNNBase import CNNBase
import os
import pickle
import pandas as pd

tf.flags.DEFINE_string("training_data_file", "inputs/train.csv", "data source for training data")
tf.flags.DEFINE_string("test_data_file", "inputs/fixed_test.csv", "data source for test data")
tf.flags.DEFINE_string("filter_sizes", '3,4,5', "comma-separated filter sizes")
tf.flags.DEFINE_integer("num_filters", 256, "number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "dropout keep probability")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda")
tf.flags.DEFINE_integer("batch_size", 1024, "batch_size")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)

##load processed inputs
f = open("processed_inputs.txt", "rb")
_ = pickle.load(f)
_ = pickle.load(f)
_ = pickle.load(f)
word_embeddings = pickle.load(f)
word2id = pickle.load(f)
f.close()

# data = pd.read_csv(FLAGS.test_data_file, low_memory=False)

# question1 = data["question1"].fillna("_na_").values
# question2 = data["question2"].fillna("_na_").values
# test_id = data["test_id"].values

# print (data.head())
# print ("the number of test samples: ", len(question1))
# sentence_num = len(question1)
# sequence_length = 60
# question1_id = np.zeros([sentence_num, sequence_length], dtype=int)
# question2_id = np.zeros([sentence_num, sequence_length], dtype=int)
# for i in range(len(question1)):
#   cleaned_q1 = clean_str(str(question1[i]))[:sequence_length]
#   cleaned_q2 = clean_str(str(question2[i]))[:sequence_length]
#   question1_id[i][:len(cleaned_q1)] = [word2id.get(w, word2id["UNK"]) for w in cleaned_q1]
#   question2_id[i][:len(cleaned_q2)] = [word2id.get(w, word2id["UNK"]) for w in cleaned_q2]

# print ("test data loaded")
# fw = open('processed_test.txt','wb')
# pickle.dump(question1_id, fw)
# pickle.dump(question2_id, fw)  
# pickle.dump(test_id, fw)
# fw.close()




fw = open("processed_test.txt", "rb")
s1 = pickle.load(fw)
s2 = pickle.load(fw)
test_id = pickle.load(fw)
fw.close()
sequence_length = len(s1[0])
num_classes = 2


cnn = CNNBase(filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))), num_filters = FLAGS.num_filters, sequence_length = sequence_length, 
                 num_classes = num_classes, pretrained_embeddings = word_embeddings, l2_reg_lamda = FLAGS.l2_reg_lambda)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver(tf.global_variables())
  ckpt = tf.train.get_checkpoint_state("runs/1534408448/checkpoints")
  if ckpt and ckpt.model_checkpoint_path:
    print ("model path: %s"%ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    print ("the number of test samples: ", len(s1))
    
    predictions = np.zeros((len(s1)))
    data = list(zip(s1, s2))
    batches = batch_iter(data, FLAGS.batch_size, shuffle = False)
    for i, batch in enumerate(batches):
      s1_batch, s2_batch = zip(*batch)
      feed = {
                cnn.input1: s1_batch,
                cnn.input2: s2_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }

      prediction = sess.run(cnn.prob, feed_dict = feed)
      end_index = min(len(s1), (i+1)*FLAGS.batch_size)
      predictions[i*FLAGS.batch_size:end_index] = prediction[:,1]
      
      print ("processing batch {:d}- samples {:d}".format(i, end_index))

  sample_submission = pd.read_csv("inputs/sample_submission.csv")
  sample_submission["is_duplicate"] = predictions
  sample_submission.to_csv("sample_submission.csv", index=False)  
  # raw_data = {"test_id":test_id, "is_duplicate":predictions}
  # df = pd.DataFrame(raw_data, columns = ["test_id", "is_duplicate"])
  # fn = 'sample_submission.csv'
  # df.to_csv(fn, index=False)  

