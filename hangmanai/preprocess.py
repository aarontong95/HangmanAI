from itertools import combinations
import torch

from hangmanai.config import MAX_LEN, COVER_SYMBOL, PAD_VALUE, COVER_VALUE, IGNORE_INDEX, BATCH_SIZE, COVER_SYMBOL

def char_to_id(char, value):
    return value if char==COVER_SYMBOL else ord(char)-97

def output_to_char(idx):
    return chr(97+idx)

def preprocess(word, pad_value, cover_value):
    _id = []
    n_word = len(word)
    for i in range(n_word):
        _id.append(char_to_id(word[i], cover_value))

    # padding
    _id += [pad_value]*(MAX_LEN - n_word)
    return _id

def preprocess_feature(word):
    return preprocess(word, PAD_VALUE, COVER_VALUE)

def preprocess_label(word):
    return preprocess(word, IGNORE_INDEX, IGNORE_INDEX)

def create_input_sample(data_train):
    labels = []
    features = []
    for word in data_train:
        word_unique = list(set(word))
        n_word = len(word)
        n_word_uq = len(word_unique)
        for i in range(n_word):
            comb_gen = combinations(range(n_word_uq), i+1)
            for comb in comb_gen:
                word_set = {word_unique[i] for i in comb}
                cover_word = [word[i] if word[i] not in word_set else COVER_SYMBOL for i in range(n_word)]
                label = [word[i] if word[i] in word_set else COVER_SYMBOL for i in range(n_word)]
                features.append(preprocess_feature(cover_word))
                labels.append(preprocess_label(label))

    return features, labels

def create_loader(features, label):
    features = torch.tensor(features)
    label = torch.tensor(label)
    train = torch.utils.data.TensorDataset(features, label)
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader