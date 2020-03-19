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
    
    
    
def load_data_and_labels(file):
    # load data from files
    trainingdata = list(open(file, "r", encoding='utf-8').readlines())
    trainingdata = [s.strip() for s in trainingdata]
    data = [clean(x) for x in trainingdata]
    news, label1, label2 = [], [], []
    for i, label in enumerate(data):
        news.append(data[0])
        label1.append(data[1])
        label2.append(data[2])   
