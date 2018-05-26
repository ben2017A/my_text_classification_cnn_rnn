import tensorflow as tf


class RNNConfig(object):
    embedding_dim = 64
    hidden_dim = 256
    num_layers = 2
    num_classes = 10
    vocab_size = 5000
    seq_length = 600
    learning_rate = 1e-3
    dropout_keep_rate = 0.5

    rnn = 'gru'

    print_per_batch = 100
    num_epochs = 10


class TextRNN():
    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length])
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        self.rnn()


    def rnn(self):
        # 设置rnn结构
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout_cell():
            if self.config.rnn == 'gru':
                cell = gru_cell()
            else:
                cell = lstm_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # 先词向量化
        embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
        embedding_input = tf.nn.embedding_lookup(embedding, self.input_x)

        # rnn单元,多层rnn神经网络
        cells = [dropout_cell() for x in range(self.config.num_layers)]
        rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        _output, _ = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=embedding_input, dtype=tf.float32)
        last = _output[:, -1, :]

        # fc + dropout + activation
        fc = tf.layers.dense(last, self.config.hidden_dim)
        fc = tf.nn.dropout(fc, self.keep_prob)
        fc = tf.nn.relu(fc)

        # 分类
        self.logits = tf.layers.dense(fc, self.config.num_classes)
        y_pred_cls = tf.argmax(self.logits, axis=1)

        # 错误
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
        self.loss = tf.reduce_mean(cross_entropy)

        # 优化
        self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        # 准确率
        accurate_pred = tf.equal(tf.argmax(self.input_y, axis=1), y_pred_cls)
        self.acc = tf.reduce_mean(tf.cast(accurate_pred, tf.float32))


