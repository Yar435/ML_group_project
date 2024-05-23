# -*- coding: utf-8 -*-
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

"""# Music genre prediction

**Описание задачи**

Вы сотрудник Отдела Data Science популярного музыкального стримингового сервиса. Сервис расширяет работу с новыми артистами и музыкантами, в связи с чем возникла задача -- правильно классифицировать новые музыкальные треки, чтобы улучшить работу рекомендательной системы. Ваши коллеги из отдела работы со звуком подготовили датасет, в котором собраны некоторые характеристики музыкальных произведений и их жанры. Ваша задача - разработать модель, позволяющую классифицировать музыкальные произведения по жанрам.


## Участники проекта, репозиторий

Участники: Храменков Я.С.

Репозиторий: [Github ML_Group_project](https://github.com/Yar435/ML_group_project/upload/main)

"""

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

train_filtered = train[train['duration_ms'] != -1.0]
test_filtered = test[test['duration_ms'] != -1.0]

duration_plus_median_train = train_filtered['duration_ms'].median()
duration_plus_median_trtest = test_filtered['duration_ms'].median()

train['duration_ms'].mask(train['duration_ms'] == -1.0, duration_plus_median_train, inplace=True)
test['duration_ms'].mask(test['duration_ms'] == -1.0, duration_plus_median_trtest, inplace=True)

train.drop(columns=['obtained_date'], inplace=True)
test.drop(columns=['obtained_date'], inplace=True)

X = train.drop(columns=['instance_id', 'music_genre'])
y = train['music_genre']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

X_test = test.drop(columns=['instance_id'])

train

"""## Разведочный анализ"""
"""**Анализ целевой переменной (music_genre)**"""
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(y=train['music_genre'], order=train['music_genre'].value_counts().index, ax=ax)
ax.set_title('Распределение музыкальных жанров')
st.pyplot(fig)

"""**Анализ распределения числовых данных**"""
numerical_features = ['acousticness', 'track_name', 'danceability',
                      'energy', 'duration_ms', 'instrumentalness',
                      'liveness', 'loudness', 'speechiness', 'tempo',
                      'valence', 'contains_ideograph']

fig, axes = plt.subplots(6, 2, figsize=(16, 10))
axes = axes.flatten()
for ax, feature in zip(axes, numerical_features):
    ax.hist(train[feature], bins=30)
    ax.set_title(feature)
plt.tight_layout()
st.pyplot(fig)

"""**Корреляционная матрица**"""
fig, ax = plt.subplots(figsize=(12, 8))
corr_matrix = train[numerical_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
ax.set_title('Корреляционная матрица числовых признаков')
st.pyplot(fig)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['acousticness', 'track_name', 'danceability', 'energy', 'duration_ms', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'contains_ideograph']),
        ('cat', categorical_transformer, [ 'trackname_language'])
    ])

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

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_val)

"""## Оценка качества"""

accuracy = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred, target_names=train['music_genre'].unique(), output_dict=True)
df_report = pd.DataFrame(report).transpose()
st.write("### Classification Report")
st.dataframe(df_report)

"""## Анализ важности признаков модели"""

model = model_pipeline.named_steps['classifier']

transformed_feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out(input_features=model_pipeline.feature_names_in_)

feature_importances = model.feature_importances_

features = transformed_feature_names
importance_df = pd.DataFrame({
    'features': features,
    'importance': feature_importances
})

fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(x='importance', y='features', data=importance_df, ax=ax)
ax.set_xlabel("Важность признаков")
ax.set_ylabel("Признаки")
ax.set_title("Важность признаков по данным модели")
st.pyplot(fig)
