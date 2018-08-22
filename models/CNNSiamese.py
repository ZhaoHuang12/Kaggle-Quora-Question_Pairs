import tensorflow as tf 
import numpy as np 

class CNNSiamese(object):
  """
  A CNN for text classification.
  Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
  """
  def __init__(self, filter_sizes, num_filters, sequence_length, num_classes, pretrained_embeddings, l2_reg_lamda=0.0):


    # Placeholders for input, output and dropout
    self.input1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
    self.input2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
    self.y = tf.placeholder(tf.int32, [None, num_classes], name="y")
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    # Keeping track of l2 regularization loss (optional)
    l2_loss = tf.constant(0.0)

    # Embedding layer
    vocab_size = pretrained_embeddings.shape[0]
    embedding_size = pretrained_embeddings.shape[1]

    with tf.name_scope("embedding"):
      glove_emdedding_initializer = tf.constant_initializer(pretrained_embeddings)
      word_embeddings = tf.get_variable(name = "word_embeddings", shape = (vocab_size, embedding_size), 
                                       initializer = glove_emdedding_initializer, trainable = True)
      self.embedded_input1 = tf.nn.embedding_lookup(word_embeddings, self.input1)#[batch_size, sequence_len, embed_size]
      self.embedded_input2 = tf.nn.embedding_lookup(word_embeddings, self.input2)

      embedded_x1 = tf.expand_dims(self.embedded_input1, -1)#[batch_size, sequence_len，embed_size,1]
      embedded_x2 = tf.expand_dims(self.embedded_input2, -1)#[batch_size, sequence_len，embed_size,1]

    
    #conv-relu-pooling-dropout  two inputs share one network
    r1 = self.create_tower(embedded_x1, sequence_length, embedding_size, filter_sizes, num_filters, reuse = False)
    r2 = self.create_tower(embedded_x2, sequence_length, embedding_size, filter_sizes, num_filters, reuse = True)
    # Final (unnormalized) scores and predictions
    with tf.name_scope("output"):
      features = tf.concat([r1, r2, tf.abs(r1 - r2), tf.multiply(r1, r2)], 1)
      num_filters_total = num_filters * len(filter_sizes)
      feature_length = 4 * num_filters_total

      num_hidden = int(np.sqrt(feature_length))
      W3= tf.get_variable(
          "W3",
          shape=[feature_length, num_hidden],
          initializer=tf.contrib.layers.xavier_initializer())
      b3 = tf.Variable(tf.constant(0.1, shape=[num_hidden]), name="b3")
      H = tf.nn.relu(tf.nn.xw_plus_b(features, W3, b3, name="hidden"))

      W4 = tf.get_variable(
          "W4",
          shape=[num_hidden, num_classes],
          initializer=tf.contrib.layers.xavier_initializer())
      b4 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b4")

      l2_loss += tf.nn.l2_loss(W4)
      l2_loss += tf.nn.l2_loss(b4)
      self.scores = tf.nn.xw_plus_b(H, W4, b4, name="scores")
      self.predictions = tf.argmax(self.scores, 1, name="predictions")

    # CalculateMean cross-entropy loss
    with tf.name_scope("loss"):
      losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.y)
      self.loss = tf.reduce_mean(losses) + l2_reg_lamda * l2_loss

    # Accuracy
    with tf.name_scope("accuracy"):
      self.y_truth = tf.argmax(self.y, 1, name="y_truth")
      self.correct_predictions = tf.equal(self.predictions, self.y_truth, name="correct_predictions")
      self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")

    with tf.name_scope("optimize"):
      self.global_step = tf.Variable(0, name="global_step", trainable=False)
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      self.grads_and_vars = optimizer.compute_gradients(self.loss)
      self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)


  def create_tower(self, embeddings, sequence_length, embedding_size, filter_sizes, num_filters, reuse):
  
    with tf.variable_scope("inference", reuse=reuse):
      # Create a convolution + maxpool layer for each filter size
      pooled_outputs = []
      for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % (filter_size)):
          # Convolution Layer
          filter_shape = [filter_size, embedding_size, 1, num_filters]
          W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
          b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
          conv = tf.nn.conv2d(
              embeddings,
              W,
              strides=[1, 1, 1, 1],
              padding="VALID",
              name="conv")

          # Apply nonlinearity
          h = tf.nn.relu(tf.nn.bias_add(conv, b), name="h")

          # Maxpooling over the outputs
          pooled = tf.nn.max_pool(
              h,
              ksize=[1, sequence_length - filter_size + 1, 1, 1],
              strides=[1, 1, 1, 1],
              padding='VALID',
              name="pool")
          pooled_outputs.append(pooled)

      # Combine all the pooled features
      num_filters_total = num_filters * len(filter_sizes)
      h_pool = tf.concat( pooled_outputs, 3)
      h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

      # Add dropout
      with tf.name_scope("dropout" ):
        h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
    return h_drop

