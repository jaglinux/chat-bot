# -*- coding: utf-8 -*-
import nltk
import numpy as np

#nltk.download('punkt')
stemmer = nltk.stem.porter.PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(words, all_words):
    result = np.zeros(len(all_words), dtype=np.float32)
    for word in words:
        word = stem(word)
        if word in all_words:
            index = all_words.index(word)
            result[index] = 1
    return result
#a = tokenize('organize organizes organizing')
#for word in a:
#    print(stem(word))

