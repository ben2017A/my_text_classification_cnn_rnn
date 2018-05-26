import numpy as np
from collections import Counter
import tensorflow.contrib.keras as kr


def read_file(filename):
    with open(filename) as f:
        labels, content = [], []
        for line in f.readlines():
            label, sentence = line.strip().split('\t')
            labels.append(label)
            content.append(list(sentence))
    return labels, content


def build_vocab(train_path, vocab_path, vocab_size):
    _, content = read_file(train_path)
    all_data = []
    for sentence in content:
        all_data.extend(sentence)
    count = Counter(all_data)
    count_pairs = count.most_common(vocab_size-1)
    words, _ = list(zip(*count_pairs))
    with open(vocab_path, 'w') as f:
        f.write('\n'.join(list(words)) + '\n')


# 读入字典，并形成{word: id}对
def read_vocab(vocab_path):
    with open(vocab_path) as f:
        words = f.read().split('\n')
        word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def read_categories():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id


def preprocess(filename, word_to_id, cat_to_id, max_length=600):
    labels, content = read_file(filename)
    x_data, y_data = [], []
    for i in range(len(content)):
        y_data.append(cat_to_id[labels[i]])
        x_data.append([word_to_id[x] for x in content[i] if x in word_to_id])
    x_pad = kr.preprocessing.sequence.pad_sequences(x_data, max_length)
    y_pad = kr.utils.to_categorical(y_data, num_classes=len(cat_to_id))
    return x_pad, y_pad


def batch_iter(x_data, y_data, batch_size=64):
    len_data = len(x_data)
    num_batch = int((len_data-1)/batch_size) + 1

    indices = np.random.permutation(np.arange(len_data))
    x_data = x_data[indices]
    y_data = y_data[indices]
    for i in range(num_batch):
        start_id = i*batch_size
        end_id = min((i+1)*batch_size, len_data)
        yield x_data[start_id:end_id], y_data[start_id:end_id]



