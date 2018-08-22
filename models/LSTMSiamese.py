import tensorflow as tf 

class SiameseLSTM(object):
  def __init__(self, rnn_size, num_layers, num_classes, pretrained_embeddings, l2_reg_lambda=0.0):
    vocab_size = pretrained_embeddings.shape[0]
    embedding_size = pretrained_embeddings.shape[1]
    l2_loss = tf.constant(0.0)
    with tf.name_scope("placeholder"):
      self.input1 = tf.placeholder(tf.int32, [None, None], name="input_x1")
      self.input2 = tf.placeholder(tf.int32, [None, None], name="input_x2")
      self.input1_length = tf.placeholder(tf.int32, [None], name='input1_length')
      self.input2_length = tf.placeholder(tf.int32, [None], name='input2_length')
      self.y = tf.placeholder(tf.int32, [None, num_classes], name="y")
      self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
      self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    with tf.name_scope("embedding"):
      glove_emdedding_initializer = tf.constant_initializer(pretrained_embeddings)
      word_embeddings = tf.get_variable(name = "word_embeddings", shape = (vocab_size, embedding_size), 
                                       initializer = glove_emdedding_initializer, trainable = True)
      self.embedded_input1 = tf.nn.embedding_lookup(word_embeddings, self.input1)#[batch_size, sequence_len, embed_size]
      self.embedded_input2 = tf.nn.embedding_lookup(word_embeddings, self.input2)

    r1 = self.LSTM(self.embedded_input1, self.input1_length, rnn_size, num_layers, self.dropout_keep_prob, reuse=False)
    r2 = self.LSTM(self.embedded_input2, self.input2_length, rnn_size, num_layers, self.dropout_keep_prob, reuse=True)

    with tf.name_scope('output'):
      features = tf.concat([r1, r2, tf.abs(r1 - r2), tf.multiply(r1, r2)], 1)
      print (features.shape)
      feature_length = rnn_size*2*4
      num_hidden1 = 256
      W3= tf.get_variable("W3",
                          shape=[feature_length, num_hidden1],
                          initializer=tf.contrib.layers.xavier_initializer())
      b3 = tf.Variable(tf.constant(0.1, shape=[num_hidden1]), name="b3")
      H3 = tf.nn.relu(tf.nn.xw_plus_b(features, W3, b3, name="hidden"))


      W4 = tf.get_variable(
                          "W4",
                          shape=[num_hidden1, num_classes],
                          initializer=tf.contrib.layers.xavier_initializer())
      b4 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b4")

      l2_loss += tf.nn.l2_loss(W4)
      l2_loss += tf.nn.l2_loss(b4)
      self.scores = tf.nn.xw_plus_b(H3, W4, b4, name="scores")
      
    with tf.name_scope("loss"):
      losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.y)
      self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
    with tf.name_scope("accuracy"):
      self.prob = tf.nn.softmax(self.scores, name='probability')
      self.predictions = tf.argmax(self.prob, 1, name="predictions")
      corrects = tf.equal(tf.argmax(self.y, 1), self.predictions)
      self.accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32), name='accuracy')

    with tf.name_scope("optimize"):
      self.global_step = tf.Variable(0, name="global_step", trainable=False)
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      self.grads_and_vars = optimizer.compute_gradients(self.loss)
      self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)
  


  def LSTM(self, inputs, inputs_length, rnn_size, num_layers, dropout, reuse):
    with tf.variable_scope("inference", reuse = reuse):
      with tf.name_scope("lstm"):
        cell = tf.contrib.rnn.BasicLSTMCell(num_units = rnn_size)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = dropout)
        cells = tf.contrib.rnn.MultiRNNCell([cell for _ in range(num_layers)])

        outputs,states = tf.nn.bidirectional_dynamic_rnn(cell_fw = cells, cell_bw = cells,
                                                     inputs = inputs, sequence_length = inputs_length,
                                                     dtype = tf.float32)
        outputs_fw, outputs_bw = outputs #[num_layers,batch_size, max_time, rnn_size]
        states_fw, states_bw = states # [num_layers,2, batch_size, rnn_size]  states are only final states on the final time step
        states_fw_c, states_fw_h = states_fw[-1] #[ batch_size, rnn_size] 取最后一层rnn的结果
        states_bw_c, states_bw_h = states_bw[-1] # [batch_size, rnn_size]

        states_concat = tf.concat([states_fw_h, states_bw_h], 1) # [num_layers, batch_size, rnn_size*2]
        return states_concat




if __name__ == "__main__":
  import numpy as np 
  pretrained_embeddings = np.zeros((4000, 200))
  bi_lstm =SiameseLSTM(rnn_size =200, num_layers = 3, num_classes = 2, 
                      pretrained_embeddings = pretrained_embeddings)
  # #test code only
  # # Create input data

  # X = np.random.randn(3,10,8) # [batch_size=3, sequence_len=10, embed_size=8]

  # # The second example is of length 6 
  # X[1,6:] = 0
  # X[2,8:] = 0

  # X_lengths = [10,6, 8]

  # cell = tf.nn.rnn_cell.LSTMCell(num_units=20)

  # outputs,states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell,cell_bw=cell,dtype=tf.float64,sequence_length=X_lengths,inputs=X)

  # output_fw, output_bw = outputs #[batch_size, sequence_len, rnn_size]
  # states_fw, states_bw = states  #[2, batch_size, rnn_size]
  # states_fw_c, states_fw_h = states_fw # [batch_size, rnn_size]
  # states_bw_c, states_bw_h = states_bw # [batch_size, rnn_size]
  # states_h =tf.concat([states_fw_h, states_bw_h], 1)
  # with tf.Session() as sess:
  #   sess.run(tf.global_variables_initializer());
  #   print(sess.run(states_fw_h))


