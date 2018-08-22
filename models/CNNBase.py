import tensorflow as tf 


class CNNBase():
 # A base cnn model for text similarity detection
 # the embedding vectors of two input sentences are concanated before fed to cnn+max_pool
 def __init__(self, filter_sizes, num_filters, sequence_length, num_classes, pretrained_embeddings, l2_reg_lamda = 0.0):

  self.pretrained_embeddings = pretrained_embeddings
  self.vocab_size = pretrained_embeddings.shape[0]
  self.embedding_size = pretrained_embeddings.shape[1]
  self.l2_loss = tf.constant(0.0)
  self.l2_reg_lamda = l2_reg_lamda

  with tf.name_scope("placeholder"):
    self.input1 = tf.placeholder(tf.int32, [None, sequence_length], name = "question1")
    self.input2 = tf.placeholder(tf.int32, [None, sequence_length], name = "question2")
    self.y = tf.placeholder(tf.int32, [None,num_classes], name = "y")
    self.learning_rate = tf.placeholder(tf.float32, name = "learning_rate")
    self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
    

  with tf.name_scope("embedding"):
    glove_emdedding_initializer = tf.constant_initializer(self.pretrained_embeddings)
    word_embeddings = tf.get_variable(name = "word_embeddings", shape = (self.vocab_size, self.embedding_size), 
                                     initializer = glove_emdedding_initializer, trainable = True)
    self.embedded_input1 = tf.nn.embedding_lookup(word_embeddings, self.input1)#[batch_size, sequence_len, embed_size]
    self.embedded_input2 = tf.nn.embedding_lookup(word_embeddings, self.input2)

    # [batch_size, sequence_len*2, embed_size]
    self.embedded_input = tf.concat([self.embedded_input1, self.embedded_input2], axis = 1)
    self.embedded_input = tf.expand_dims(self.embedded_input, -1)#[batch_size, sequence_len2，embed_size,1]

  pooled_output = []
  for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool-%s" %filter_size):
      filter_shape = [filter_size, self.embedding_size, 1, num_filters]
      W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name = 'W')
      B = tf.Variable(tf.constant(0.1, shape = [num_filters]), name = 'B')
      conv = tf.nn.conv2d(self.embedded_input, W, strides=[1,1,1,1], padding="VALID", name='conv')
      h = tf.nn.relu(tf.nn.bias_add(conv,B), name = 'relu')
      # [batch_size, sequence_len*2-filter_size+1, 1, num_filters]

      #max-pooling over the outputs, result dimension [batch_size, 1, 1, num_filters]
      pooled = tf.nn.max_pool(h, ksize=[1, sequence_length*2-filter_size+1,1,1],
                             strides = [1,1,1,1], padding="VALID", name = "pool")
      
      pooled_output.append(pooled)

  #combine alll pooled features
  num_filters_total = num_filters*len(filter_sizes)
  h_pooled = tf.concat(pooled_output, 3) #[batch_size, 1, ,1 num_filters_total]
  h_pooled_flat = tf.reshape(h_pooled, [-1, num_filters_total])# [batch_size, num_filters_total]
  
  #dropout
  with tf.name_scope("dropout"):
    h_drop = tf.nn.dropout(h_pooled_flat, self.dropout_keep_prob)

  with tf.name_scope("output"):
    w = tf.get_variable("w", shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
    self.l2_loss += tf.nn.l2_loss(w)
    self.l2_loss += tf.nn.l2_loss(b)
    self.scores = tf.nn.xw_plus_b(h_drop, w, b, name='scores')

  with tf.name_scope("loss"):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits = self.scores, labels = self.y)
    self.loss = tf.reduce_mean(losses)+self.l2_reg_lamda*self.l2_loss

  with tf.name_scope("accuracy"):
    self.prob = tf.nn.softmax(self.scores, name = 'probalbility')
    self.predictions = tf.argmax(self.prob, 1, name = 'predictions')
    self.correct = tf.equal(self.predictions, tf.argmax(self.y, 1))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32), name='accuracy')
    
  with tf.name_scope("optimize"):
    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    self.grads_and_vars = optimizer.compute_gradients(self.loss)
    self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)
  
