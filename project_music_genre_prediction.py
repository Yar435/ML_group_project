# -*- coding: utf-8 -*-
"""project_music_genre_prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1auyy-oDiM6f5r-dXBablCcuRy7Z9EQdI

# Music genre prediction

**Описание задачи**

Вы сотрудник Отдела Data Science популярного музыкального стримингового сервиса. Сервис расширяет работу с новыми артистами и музыкантами, в связи с чем возникла задача -- правильно классифицировать новые музыкальные треки, чтобы улучшить работу рекомендательной системы. Ваши коллеги из отдела работы со звуком подготовили датасет, в котором собраны некоторые характеристики музыкальных произведений и их жанры. Ваша задача - разработать модель, позволяющую классифицировать музыкальные произведения по жанрам.

В ходе работы пройдите все основные этапы полноценного исследования:

*  загрузка и ознакомление с данными
*  предварительная обработка
*  полноценный разведочный анализ
*  разработка новых синтетических признаков
*  проверка на мультиколлинеарность
*  отбор финального набора обучающих признаков
*  выбор и обучение моделей
*  итоговая оценка качества предсказания лучшей модели
*  анализ важности ее признаков

**ВАЖНО**  
Необходимо реализовать решение с использованием технологии `pipeline` (из библиотеки `sklearn`)

**ОЖИДАЕМЫЙ РЕЗУЛЬТАТ**

* Оформленный репозиторий на GitHub (ноутбук с исследованием + код приложения)
* Развернутое web-приложение (с использованием библиотеки Streamlit)

## Участники проекта, репозиторий, приложение

Участники: Храменков Я.С.

Репозиторий: https://github.com/Yar435/ML_group_project/upload/main

## Импорт библиотек, установка констант
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import re
from langdetect import detect, LangDetectException
import os
import streamlit as st

TRAIN = "https://www.dropbox.com/scl/fi/5zy935lqpaqr9lat76ung/music_genre_train.csv?rlkey=ccovu9ml8pfi9whk1ba26zdda&dl=1"
TEST = "https://www.dropbox.com/scl/fi/o6mvsowpp9r3k2lejuegt/music_genre_test.csv?rlkey=ac14ydue0rzlh880jwj3ebum4&dl=1"

RANDOM_STATE = 42
TEST_SIZE = 0.10

"""## Загрузка и обзор данных"""

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

train

"""**Описание полей данных**

`instance_id` - уникальный идентификатор трека  
`track_name` - название трека  
`acousticness` - акустичность  
`danceability` - танцевальность  
`duration_ms` -продолжительность в милисекундах  
`energy` - энергичность  
`instrumentalness` - инструментальность  
`key` - тональность  
`liveness` - привлекательность  
`loudness` - громкость  
`mode` - наклонение  
`speechiness` - выразительность  
`tempo` - темп  
`obtained_date` - дата загрузки в сервис  
`valence` - привлекательность произведения для пользователей сервиса  
`music_genre` - музыкальный жанр

"""

language = pd.read_csv('lang_503921.csv')
train['trackname_language'] = language['trackname_language']

regex = re.compile('[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+')

def contains_ideograph(text):
    bool_num = bool(regex.search(text))
    if bool_num:
        return 1
    else:
        return 0

train['contains_ideograph'] = train['track_name'].apply(contains_ideograph)
test['contains_ideograph'] = test['track_name'].apply(contains_ideograph)

train['track_name'] = train['track_name'].apply(lambda x: len(x))
test['track_name'] = test['track_name'].apply(lambda x: len(x))

# избавляемся от выбросов в duration_ms (-1.0)

train_filtered = train[train['duration_ms'] != -1.0]
test_filtered = test[test['duration_ms'] != -1.0]

duration_plus_median_train = train_filtered['duration_ms'].median()
duration_plus_median_trtest = test_filtered['duration_ms'].median()

train['duration_ms'].mask(train['duration_ms'] == -1.0, duration_plus_median_train, inplace=True)
test['duration_ms'].mask(test['duration_ms'] == -1.0, duration_plus_median_trtest, inplace=True)

# Удалим ненужный столбец 'obtained_date', так как он не добавляет полезной информации для предсказания жанра
train.drop(columns=['obtained_date'], inplace=True)
test.drop(columns=['obtained_date'], inplace=True)

# Разделим тренировочные данные на тренировочную и валидационную выборки
X = train.drop(columns=['instance_id', 'music_genre'])
y = train['music_genre']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

# Подготовим тестовые данные(для использования в будущем)
X_test = test.drop(columns=['instance_id'])

train

"""## Разведочный анализ"""
"""**Анализ целевой переменной (music_genre)**"""
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(y=train['music_genre'], order=train['music_genre'].value_counts().index, ax=ax)
ax.set_title('Распределение музыкальных жанров')
st.pyplot(fig)

"""**Анализ рапределения числовых данных**"""
numerical_features = ['acousticness', 'track_name', 'danceability',
                      'energy', 'duration_ms', 'instrumentalness',
                      'liveness', 'loudness', 'speechiness', 'tempo',
                      'valence', 'contains_ideograph']

plt.figure(figsize=(16, 10))
train[numerical_features].hist(bins=30, figsize=(16, 10), layout=(6, 2))
plt.tight_layout()
st.pyplot(plt.show())

"""**Корреляционная матрица**"""
plt.figure(figsize=(12, 8))
corr_matrix = train[numerical_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Корреляционная матрица числовых признаков')
st.pyplot(plt.show())


# Преобразователь для числовых признаков: заполняем пропуски средним значением и масштабируем
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Преобразователь для категориальных признаков: заполняем пропуски значением моды и применяем One-Hot Encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Объединяем обработку для разных типов признаков
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['acousticness', 'track_name', 'danceability', 'energy', 'duration_ms', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'contains_ideograph']),
        ('cat', categorical_transformer, [ 'trackname_language'])
    ])



# Создаем конвейер, включающий предварительную обработку и обучение модели
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=4,
    max_features='sqrt',
    random_state=42
))
])

# Обучение модели
model_pipeline.fit(X_train, y_train)

# Прогнозирование
y_pred = model_pipeline.predict(X_val)

# os.rename("lang.csv", "lang_"+ str(accuracy*1000000).split('.')[0] +".csv")

"""## Оценка качества"""

# Оценка точности
accuracy = accuracy_score(y_val, y_pred)
# print(f"Точность: {accuracy:.4f}")
report = classification_report(y_val, y_pred)
df_report = pd.DataFrame(report).transpose()

# Отображение отчета в Streamlit
st.write("### Classification Report")
st.dataframe(df_report)


"""## Анализ важности признаков модели"""

# Доступ к модели
model = model_pipeline.named_steps['classifier']

transformed_feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out(input_features=model_pipeline.feature_names_in_)

# Получение важности признаков
feature_importances = model.feature_importances_

# Визуализация для случая без полиномиальных признаков
features = transformed_feature_names
importance_df = pd.DataFrame({
    'features': features,
    'importance': feature_importances
})

"""**Визуализация важности признаков**"""
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='features', data=importance_df)
plt.xlabel("Важность признаков")
plt.ylabel("Признаки")
st.pyplot(plt.show())

