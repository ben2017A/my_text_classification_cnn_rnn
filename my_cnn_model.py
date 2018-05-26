import tensorflow as tf


class CNNConfig(object):
    # 配置CNN模型
    embedding_dim = 64
    seq_length = 600
    vocab_size = 5000
    num_filters = 256
    kernel_size = 5
    num_classes = 10
    hidden_dim = 128 # 全连接层长度

    learning_rate = 1e-3
    dropout_keep_prob = 0.5

    num_epochs = 10
    print_per_batch = 100


class TextCNN():
    def __init__(self, config):
        self.config = config

        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length])
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        self.cnn()


    def cnn(self):
        # 开始构建cnn结构
        # 词嵌入层
        embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
        embedding_input = tf.nn.embedding_lookup(embedding, self.input_x)

        # 卷积层
        conv = tf.layers.conv1d(embedding_input, self.config.num_filters, self.config.kernel_size)
        # pooling层
        gmp = tf.reduce_max(conv,reduction_indices=[1])

        # fc + dropout + activation
        fc = tf.layers.dense(gmp, self.config.hidden_dim)
        fc = tf.nn.dropout(fc, self.keep_prob)
        fc = tf.nn.relu(fc)

        # 预测类别
        logits = tf.layers.dense(fc, self.config.num_classes)
        self.y_pred_cls = tf.argmax(tf.nn.softmax(logits), axis=1) # 预测类别

        # 计算错误
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_y)
        self.loss = tf.reduce_mean(cross_entropy)

        # 优化
        self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        # 计算准确率
        correct_pred = tf.equal(tf.argmax(self.input_y, axis=1),self.y_pred_cls)
        self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))





