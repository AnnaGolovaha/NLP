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
print(data.shape) #(6307,34)
n = data.shape[0] - inp_chars #6301 (без последних 6 символов текста)
print(n)

# создаёт массив массивов по 6 строк
# Т.е. каждую итерацию мы спускаемся на один символ
# сначала записываем массив OHE векторов для первых 6 символов, затем для 6 символов, но начиная со 2-го в тексте и тд
# получим 6301 массив по 6 векторов(строк) в каждом
X = np.array([data[i:i+inp_chars, :] for i in range(n)]) #входные данные
Y = data[inp_chars:] #предсказываемые символы (начинаем предсказывать с 7-го)

model = Sequential()
# (размер батча, число символов, размер векторов)
# в данном случае размер батча автоматически рассчитывается
model.add(Input((inp_chars, num_characters)))
model.add(SimpleRNN(128, activation = 'tanh')) #Рекуррентный слой на 128 нейронов
model.add(Dense(num_characters, activation = 'softmax')) #полносвязный слой из 34-х нейронов
model.summary()

model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer = 'adam')
history = model.fit(X,Y, batch_size = 32, epochs = 100)

# Создаём функцию, которая будет строить фразу на основе прогнозируемых значений
def buildPhrase(inp_str, str_len = 50):
    for i in range(str_len): #идём по заданной длине строки (в конце получим 50 символов)
        x = []
        #берём первые 6 символов (изначально задаём строку из 6-ти символов)
        for j in range(i, i + inp_chars):
            x.append(tokenizer.texts_to_matrix(inp_str[j])) #символ преобразуем в OHE
        
        x = np.array(x)
        inp = x.reshape(1, inp_chars, num_characters) #тот же х, только правильно записанный (6 строк по 34 символа)

        pred = model.predict(inp) #Предсказываем OHE вектор 7-го символа по 6 предыдущим
        d = tokenizer.index_word[pred.argmax(axis=1)[0]] #переводим ответ в символ из вектора

        inp_str += d #дописываем строку
    
    return inp_str

res = buildPhrase('утренн')
print(res)