import sys
import csv
import os
import pandas as pd
import re
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout, Flatten
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from gensim.models import Word2Vec
import jieba
import itertools
from sklearn.model_selection import train_test_split
#from tensorflow import learn


def clean(string):
//清除掉中文句子中的无意义字符
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

y_train_data = list()
labels = list()
with open('label.csv') as f:
    reader = csv.DictReader(f)
    for i in reader:
        y_train_data.append(i['Label'])
        #y_train_label2.append(i['Subtopic'])
        #print (i['Label'])
for item in y_train_data:
        if item not in labels:
            labels.append(item)
labels_len = len(labels)
print('新闻分类数为： ', labels_len)
print('新闻分类:', labels)

print('样本数:', len(y_train_data))
#categories = ['game','tech', 'edu','social','house','food',
                  #'sports','finance', 'law', 'health','travel', 
                 # 'career','ent','religion','weather','emotion', 
                 # 'baby','cul','auto','agriculture','women',
                  #'politics','abroad','inspiration','comic', 
                  #'astro','funny','digital','beauty','history']
cat_to_id = dict(zip(labels, range(len(labels))))
print(cat_to_id)
y_train_data = np.asarray(y_train_data)
y_label = np.zeros(len(y_train_data),dtype = int)
index = 0
for name in y_train_data:
    #print(name)
    y_label[index] = cat_to_id[name]
    index = index + 1

#print(y_label)


def load_data_and_labels(train_file,y_label):
    # Load data from files
    texts = list()  # 训练数据
    y_train_data = list()  # y训练分类数据
    x_test = list()  # 测试数据
    y_test_data = list()  # y测试分类数据
    y_labels = list()  # 分类集
    print("Loading data...")
    with open(train_file, 'r', encoding='utf-16') as train_file:
        for line in train_file.read().split('\n'):
            texts.append(' '.join(jieba.cut(line)))


    max_document_length = max([len(x.split(" ")) for x in texts])
    print("max_document_length", max_document_length)
  
    tokenizer = Tokenizer(num_words = 30000)
    tokenizer.fit_on_texts(texts)
    x_train, x_test, y_train, y_test = train_test_split(texts,y_label, test_size=0.1,random_state=0)
    x_train_word_ids = tokenizer.texts_to_sequences(x_train)
    x_test_word_ids = tokenizer.texts_to_sequences(x_test)
    vocab = tokenizer.word_index  # 一个dict，保存所有word对应的编号id，从1开始
    # 每条样本长度不唯一，将每条样本的长度设置一个固定值
    x_train_padded_seqs=pad_sequences(x_train_word_ids,maxlen=300) #将超过固定值的部分截掉，不足的在最前面用0填充
    x_test_padded_seqs=pad_sequences(x_test_word_ids, maxlen=300)
    
    print('Found %s unique tokens.' , len(vocab))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))
    return x_train_padded_seqs, y_train, x_test_padded_seqs, y_test, vocab
    
x_train, y_train, x_test, y_test, vocab = load_data_and_labels("NewsText.txt",y_label)


