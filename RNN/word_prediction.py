import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

from tensorflow.keras.layers import Dense, SimpleRNN, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.utils import to_categorical

with open('text.txt', 'r', encoding='utf-8') as f:
    texts = f.read()
    texts = texts.replace('\ufeff','') #убираем первый невидимый символ

maxWordsCount = 1000 #задаём максимальный размер для словаря из уникальных слов (это и будет длина OHE вектора) 
# (слов может оказаться меньше)
# num_words -выберет наиболее часто встречающиеся слова
# lower=True - переводит всё в нижний регистр
# split=' ' -разбиваем слова по пробелам
#  char_level=False - разбиваем по словам, а не по символам
tokenizer = Tokenizer(num_words = maxWordsCount, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
                      lower=True, split=' ', char_level=False)
tokenizer.fit_on_texts([texts])#формируем токены из нашего текста
#print(tokenizer.word_index) #получим словарь из слов и их индексов 

#словарь с указанием частоты каждого слова в кортежах
dist = list(tokenizer.word_counts.items()) #[('я', 21), ('притягиваю', 1), ('только', 21)
print(dist[:10])

data = tokenizer.texts_to_sequences([texts]) #здесь получим список индексов для каждого слова в тексте
# ранее мы присвоили каждому слову свой индекс 
# print(data)
res = to_categorical(data[0], num_classes = maxWordsCount) #тут получим уже OHE вектора (кол-во столбцов=1000)
print(res.shape)

inp_words = 3 #будем по 3-м словам предсказывать 4-е
n = res.shape[0] - inp_words 

X = np.array([res[i:i+inp_words,:] for i in range(n)])
Y = res[inp_words:]

model = Sequential()
model.add(Input((inp_words, maxWordsCount)))
model.add(SimpleRNN(400, activation='tanh'))
model.add(Dense(maxWordsCount, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(X, Y, batch_size=32, epochs=50)

def buildPhrase(texts, str_len = 10):
    res = texts #подали на вход 3 слова
    #[texts] - это массив входных слов
    # преобразуем в массив индексов для этих слов
    #[0] чтобы взять список из списка [[98, 520, 521]]
    data = tokenizer.texts_to_sequences([texts])[0] 
    for i in range(str_len):
        x = to_categorical(data[i:i+inp_words], num_classes = maxWordsCount) #OHE преобразование
        inp = x.reshape(1, inp_words, maxWordsCount)

        pred = model.predict(inp)
        indx = pred.argmax(axis = 1)[0]
        data.append(indx)

        res += ' ' + tokenizer.index_word[indx]

    return res

res = buildPhrase('позитив добавляет годы')
print(res)