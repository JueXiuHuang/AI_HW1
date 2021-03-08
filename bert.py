#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from IPython.display import display, HTML
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from transformers import *
from official import nlp
import official.nlp.optimization
import tensorflow as tf
import pandas as pd
import numpy as np
import re


# In[2]:


def read_data(path):
    df = pd.read_excel(path)
    # Columns: 'id','name','description','category','owner name','location',
    #          'Label1','Label2','Final_Label_Kaggle','Final_Label'
    # Keep: 'name','description','category','Final_Label_Kaggle'
    df = df.drop(['id', 'location', 'Final_Label', 'Label1', 'Label2'], axis=1)
    df['name'].fillna('', inplace=True)
    df['description'].fillna('', inplace=True)
    df['category'].fillna('', inplace=True)
    df['owner name'].fillna('', inplace=True)
    #df_useful = pd.concat([df.iloc[0:500], df.iloc[800:900]], axis=0)
    df_useful = df
    df_useful.dropna(inplace=True)
    return df_useful

def data_preprocess(df):
    df['X'] = df['name'] + df['description'] + df['category'] + df['owner name']
    df = df.drop(['name', 'description', 'category', 'owner name'], axis=1)
    
    X = list(df['X'])
    y = list(df['Final_Label_Kaggle'])
    
    # Remove nonChinese charactor
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    for x in range(len(X)):
        X[x] = re.sub(pattern, '', X[x])
    # Convert labels into binary form
    # First convert into integer form
    for i in range(len(y)):
        ints = y[i].split(' ')
        for j in range(len(ints)):
            ints[j] = int(ints[j])
        y[i] = ints
    # Convert into binary form
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y)
    return X, y

def encode_data(X, tokenizer):
    input_ids, token_type_ids, attention_mask = [], [], []
    for i in range(len(X)):
        inputs = tokenizer.encode_plus(X[i],add_special_tokens=True, max_length=384, pad_to_max_length=True,
            return_attention_mask=True, return_token_type_ids=True, truncation=True)
        input_ids.append(inputs['input_ids'])
        token_type_ids.append(inputs['token_type_ids'])
        attention_mask.append(inputs['attention_mask']) 
    return np.asarray(input_ids, dtype='int32'), np.asarray(attention_mask, dtype='int32'), np.asarray(token_type_ids, dtype='int32')

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
    return


# In[3]:


def build_model(model_base, learning_rate, epochs, batch_size, train_data_size):
    model = TFBertModel.from_pretrained(model_base, return_dict=True)
    input1 = tf.keras.Input(shape=(384,), dtype=tf.int32, name='input_ids')
    input2 = tf.keras.Input(shape=(384,), dtype=tf.int32, name='attention_mask')
    input3 = tf.keras.Input(shape=(384,), dtype=tf.int32, name='token_type_ids')
    outDict = model([input1, input2, input3])
    pooled_output = outDict['pooler_output']
    distribution = tf.keras.layers.Dense(8, activation='sigmoid', name='Output_Layer')(pooled_output)
    steps_per_epoch = int(train_data_size/batch_size)
    num_train_steps = steps_per_epoch * epochs
    warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)
    opWarm = nlp.optimization.create_optimizer(learning_rate, num_train_steps=num_train_steps,
                                                  num_warmup_steps=warmup_steps)
    model = tf.keras.Model(inputs=[input1, input2, input3], outputs=distribution)
    loss = tf.keras.losses.BinaryCrossentropy()
    metric = tf.keras.metrics.BinaryAccuracy('BinaryAccuracy')
    model.compile(optimizer=opWarm, loss=loss, metrics=[metric])
    
    model.summary()
    return model


# In[4]:


Training = read_data('TrainingData.xlsx')
#display(HTML(Training[0:10].to_html()))
X, y = data_preprocess(Training)
X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size=0.2)
model_base = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_base)
train_input_ids, train_attention_mask, train_token_type_ids = encode_data(X_train, tokenizer)
vali_input_ids, vali_attention_mask, vali_token_type_ids = encode_data(X_vali, tokenizer)


# In[5]:


label_sum = [0,0,0,0,0,0,0,0]
for l in range(len(y_train)):
    for i in range(8):
        if y_train[l][i] == 1:
            label_sum[i] = label_sum[i]+1
            
the_max = max(label_sum)
c_weight = {
    0:the_max/label_sum[0],
    1:the_max/label_sum[1],
    2:the_max/label_sum[2],
    3:the_max/label_sum[3],
    4:the_max/label_sum[4],
    5:the_max/label_sum[5],
    6:the_max/label_sum[6],
    7:the_max/label_sum[7]
}
print(c_weight)


# In[6]:


learning_rate = 3e-5
epochs = 4
batch_size = 8
train_data_size = len(y_train)
model = build_model(model_base, learning_rate, epochs, batch_size, train_data_size)


# In[7]:


history = model.fit([train_input_ids, train_attention_mask, train_token_type_ids],
                    y_train, epochs=epochs, batch_size=batch_size, class_weight=c_weight)
plot_learning_curves(history)


# In[8]:


y_pred = model.predict([vali_input_ids, vali_attention_mask, vali_token_type_ids])
y_pred = y_pred.round()
print(classification_report(y_vali, y_pred))


# In[9]:



testdf = pd.read_excel('TestingData.xlsx')
# Columns: 'id','name','description','category','owner name','location',
# Keep: 'name','description','category','id'
testdf = testdf.drop(['location'], axis=1)
testdf['name'].fillna('', inplace=True)
testdf['description'].fillna('', inplace=True)
testdf['category'].fillna('', inplace=True)
testdf['owner name'].fillna('', inplace=True)
testdf['X'] = testdf['name'] + testdf['description'] + testdf['category'] + testdf['owner name']
testdf = testdf.drop(['name', 'description', 'category', 'owner name'], axis=1)
test_X = list(testdf['X'])
ids = list(testdf['id'])
# Remove nonChinese charactor
pattern = re.compile(r'[^\u4e00-\u9fa5]')
for x in range(len(test_X)):
    test_X[x] = re.sub(pattern, '', test_X[x])
test_input_ids, test_attention_mask, test_token_type_ids = encode_data(test_X, tokenizer)
p = model.predict([test_input_ids, test_attention_mask, test_token_type_ids])
finalout = []
for i in range(len(p)):
    s = ''
    for j in range(8):
        if p[i][j] >= 0.5:
            if len(s) != 0:
                s = s + ' ' + str(j)
            else:
                s = s + str(j)
    finalout.append(s)
submit = pd.DataFrame(list(zip(ids, finalout)), columns =['Id', 'Predicted'])
display(HTML(submit.to_html()))
submit.to_csv('bertResult.csv', index=False)
print(submit.columns)


# In[ ]:




