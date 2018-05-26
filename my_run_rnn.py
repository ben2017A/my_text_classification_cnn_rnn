import time
from datetime import timedelta
import tensorflow as tf
import os
import sys
from my_load import *
from my_rnn_model import *
from sklearn import metrics

base_dir = 'data/cnews'
train_path = os.path.join(base_dir, 'cnews.train.txt')
test_path = os.path.join(base_dir, 'cnews.test.txt')
val_path = os.path.join(base_dir, 'cnews.val.txt')
vocab_path = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')


def get_time_diff(start_time):
    end_time = time.time()
    time_diff = end_time - start_time
    return timedelta(seconds=int(round(time_diff)))


def feed_data(x_data, y_data, keep_prob):
    feed_dict = {
        model.input_x: x_data,
        model.input_y: y_data,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_data, y_data):
    data_len = len(x_data)
    batch_data = batch_iter(x_data, y_data, batch_size=128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_data:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, keep_prob=1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss = total_loss + loss*batch_len
        total_acc = total_acc + acc*batch_len
    return total_acc/data_len, total_loss/data_len


def train():
    # 配置saver
    saver = tf.train.Saver()
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("loading training and testing data...")
    start_time = time.time()
    x_train, y_train = preprocess(train_path, word_to_id, cat_to_id, max_length=config.seq_length)
    x_val, y_val = preprocess(val_path, word_to_id, cat_to_id, max_length=config.seq_length)
    time_dif = get_time_diff(start_time)
    print("usage time: ", time_dif)



    # 开始构建session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print("training and evaluating...")

    start_time = time.time()
    total_batch = 0
    best_acc_val = 0.0
    last_improved = 0
    required_improvement = 1000


    flag = False
    for epoch in range(config.num_epochs):
        print("Epoch:", epoch + 1)
        batch_train = batch_iter(x_train, y_train, batch_size=128)
        for x_batch, y_batch in batch_train:
            feed_dict =feed_data(x_batch, y_batch, 1.0)

            if total_batch%config.print_per_batch==0:
                train_loss, train_acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
                val_acc, val_loss = evaluate(sess, x_val, y_val)

                if val_acc > best_acc_val:
                    best_acc_val = val_acc
                    last_improved = total_batch
                    saver.save(sess=sess, save_path=save_path)
                    improved_star = '*'
                else:
                    improved_star = ''


                time_dif = get_time_diff(start_time)
                msg = 'Iter:{0:>6}, Train loss: {1:>6.2}, Train_acc: {2:>6.2},' \
                      'val loss: {3:>6.2}, val acc:{4:>6.2}, Time:{5} {6} '
                print(msg.format(total_batch, train_loss, train_acc, val_loss, val_acc, time_dif, improved_star))

            sess.run(model.optim, feed_dict=feed_dict)
            total_batch += 1
            if total_batch - last_improved>=required_improvement:
                print("No optimazation for too long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def test():
    print("loading test data...")
    start_time = time.time()
    x_test, y_test = preprocess(test_path, word_to_id, cat_to_id, max_length=config.seq_length)

    # 从save_path中读出sess
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path=save_path)

    print("Testing...")
    test_acc, test_loss = evaluate(sess, x_test, y_test)
    msg = 'test loss: {0:>6.2}, test acc: {1:>6.2}'
    print(msg.format(test_loss, test_acc))

    # 生成更多测试集的信息，例如F1 score.因此要先生成测试集的全部y_pred_cls
    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len-1)/batch_size)

    y_test_cls = np.argmax(y_test, axis=1)
    y_pred_cls = np.zeros(shape=data_len, dtype=np.int32)
    for i in range(num_batch):
        start_id = i*batch_size
        end_id = min((i+1)*batch_size, data_len)

        batch_x = x_test[start_id:end_id]
        feed_dict = {
            model.input_x: batch_x,
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = sess.run(model.y_pred_cls, feed_dict=feed_dict)

    # 评估
    print("precision, recall and F1 score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("confusion matrix...")
    confusion_matrix = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(confusion_matrix)

    time_dif = get_time_diff(start_time)
    print("time usage: ", time_dif)


if __name__ == "__main__":
    if len(sys.argv)!=2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("usage: python run_cnn.py [train / test]")

    print("configuring CNNConfig")
    config = RNNConfig()
    if not os.path.exists(vocab_path):
        build_vocab(train_path, vocab_path,vocab_size=config.vocab_size)
    words, word_to_id = read_vocab(vocab_path)
    categories, cat_to_id = read_categories()
    config.vocab_size = len(words)

    model = TextRNN(config)
    if sys.argv[1]=='train':
        train()
    else:
        test()













