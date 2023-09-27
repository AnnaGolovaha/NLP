import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import re

from tensorflow.keras.layers import Dense, SimpleRNN, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer

with open('train_data_true.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.replace('\ufeff', '')  # убираем первый невидимый символ
    #Оставляем только символы русского алфавита от А-я и пробел, будет 34 разных символа 'А-я '
    text = re.sub(r'[^А-я ]', '', text)  # заменяем все символы кроме кириллицы на пустые символы

num_characters = 34 #33 буквы и пробел
# Архитектура many to one
# Нужно получить One-hot encoding (OHE) векторы
# На вход такой сети подаются векторы из 34-х символов вида [0,0,1,0,0...]
# 1 стоит на месте соответсвующей буквы в алфавитном порядке (для а на первом [1,0,0..])

#разбиваем на символы
 # num_words - максимальное кол-во возвращаемых символов
 # char_level = True - делит текст на символы, False - на слова
tokenizer = Tokenizer(num_words = num_characters, char_level = True)                                                                
tokenizer.fit_on_texts([text]) #формируем токены из нашего текста
print(tokenizer.word_index) #получим словарь из символов и их индексов (не по алфавиту)

inp_chars = 6 #столько векторов используется для предсказания  (6 первых символов текста, а 7 предсказываем)
data = tokenizer.texts_to_matrix(text) #получим массив OHE, в нём будет столько строк, сколько букв в тексте (6307)
print(data.shape)
n = data.shape[0] - inp_chars

#X = np.array([data[i:i+inp_chars, :] for i in range(n)])

