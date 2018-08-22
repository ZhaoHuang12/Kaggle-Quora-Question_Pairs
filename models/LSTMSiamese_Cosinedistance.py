import tensorflow as tf
import numpy as np

class SiameseLSTM(object):
  """
  A LSTM based deep Siamese network for text similarity.
  Uses an character embedding layer, followed by a biLSTM and Energy Loss layer.
  """
  
  def BiRNN(self, x, dropout, scope, embedding_size, sequence_length, hidden_units):
    n_hidden=hidden_units
    n_layers=3
    # Prepare data shape to match `static_rnn` function requirements
    x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
    print("{} * {}".format(len(x), x[0].shape))
    # Define lstm cells with tensorflow
    # Forward direction cell
    with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
      stacked_rnn_fw = []
      for _ in range(n_layers):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
        stacked_rnn_fw.append(lstm_fw_cell)
      lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

    with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
      stacked_rnn_bw = []
      for _ in range(n_layers):
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell,output_keep_prob=dropout)
        stacked_rnn_bw.append(lstm_bw_cell)
      lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
    # Get lstm cell output

    with tf.name_scope("bw"+scope),tf.variable_scope("bw"+scope):
      outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)
    return outputs[-1]
  
  def contrastive_loss(self, y,d):
    tmp= y *tf.square(d)
    #tmp= tf.mul(y,tf.square(d))
    tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
    return tf.reduce_mean(tmp +tmp2)/2
  
  def __init__(
    self, hidden_units, num_layers, sequence_length, pretrained_embeddings,l2_reg_lambda=0.0):
    #self, sequence_length, vocab_size, embedding_size, hidden_units, l2_reg_lambda, batch_size):
    embedding_size = pretrained_embeddings.shape[1]
    vocab_size = pretrained_embeddings.shape[0]
    # Placeholders for input, output and dropout
    self.input1 = tf.placeholder(tf.int32, [None, sequence_length], name="input1")
    self.input2 = tf.placeholder(tf.int32, [None, sequence_length], name="input2")
    self.y = tf.placeholder(tf.float32, [None], name="y")
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    # Keeping track of l2 regularization loss (optional)
    l2_loss = tf.constant(0.0, name="l2_loss")

      
    with tf.name_scope("embedding"):
      glove_emdedding_initializer = tf.constant_initializer(pretrained_embeddings)
      word_embeddings = tf.get_variable(name = "word_embeddings", shape = (vocab_size, embedding_size), 
                                       initializer = glove_emdedding_initializer, trainable = True)
      self.embedded_chars1 = tf.nn.embedding_lookup(word_embeddings, self.input1)#[batch_size, sequence_len, embed_size]
      self.embedded_chars2 = tf.nn.embedding_lookup(word_embeddings, self.input2)

    # Create a convolution + maxpool layer for each filter size
    with tf.name_scope("output"):
      self.out1=self.BiRNN(self.embedded_chars1, self.dropout_keep_prob, "side1", embedding_size, sequence_length, hidden_units)
      self.out2=self.BiRNN(self.embedded_chars2, self.dropout_keep_prob, "side2", embedding_size, sequence_length, hidden_units)
      self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1,self.out2)),1,keep_dims=True))
      self.distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1),1,keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(self.out2),1,keep_dims=True))))
      self.distance = tf.reshape(self.distance, [-1], name="distance")
    with tf.name_scope("loss"):
      self.loss = self.contrastive_loss(self.y,self.distance)
    #### Accuracy computation is outside of this class.
    with tf.name_scope("accuracy"):
      self.temp_sim = tf.subtract(tf.ones_like(self.distance),tf.rint(self.distance), name="temp_sim") #auto threshold 0.5
      correct_predictions = tf.equal(self.temp_sim, self.y)
      self.accuracy=tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
    with tf.name_scope("optimize"):
      self.global_step = tf.Variable(0, name="global_step", trainable=False)
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      self.grads_and_vars = optimizer.compute_gradients(self.loss)
      self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)

