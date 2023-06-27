import streamlit as st
import pandas as pd
import numpy as np
import joblib
import scipy
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

#import modeles

modele_NaiveBais = joblib.load('modele_class ification_articles.joblib')
Bert_Model=AutoModelForSequenceClassification.from_pretrained('./model')
Lstm_Model=load_model('./ModelLstm')
tokenizerLstm = Tokenizer(num_words=10000)
tokenizerBert = AutoTokenizer.from_pretrained("./model")
top1PrecentMixtureVectorizer=joblib.load('top1PrecentMixtureVectorizer.joblib')
def toNumpyArray(data):
    data_type = type(data)
    if data_type == np.ndarray:
        return data
    elif data_type == list:
        return np.array(data_type)
    elif data_type == scipy.sparse.csr_matrix:
        return data.toarray()
    print(data_type)
    return None


def NaiveBaise_Model(text):
    # vectorize the text
    test = top1PrecentMixtureVectorizer.transform([text])
    var_test = toNumpyArray(test)
    l= modele_NaiveBais.predict(var_test)
    # test_norml=normalize(var_test, norm='l2', axis=1, copy=True, return_norm=False)
    # Check for the prediction probability
    pred_proba = modele_NaiveBais.predict_proba(var_test)
    pred_percentage_for_all = dict(zip(modele_NaiveBais.classes_, pred_proba[0]))
    #print("Prediction using Logistic Regression Top 1%:  : {} , Prediction Score : {}".format(l[0], np.max(pred_proba)))
    #print(l[0])
    #print(pred_percentage_for_all)
    return "la categorie de votre article est: "+l[0]
def Model_Bert(data):
    inputs = tokenizerBert(data, return_tensors="pt")
    outputs = Bert_Model(**inputs)
    label_map = {
        0: 'سياسة',
        1: 'اقتصاد',
        2: 'صحة'}
    # Obtenir les prédictions
    predictions = outputs.logits.argmax(dim=1)  # tensor
    predicted_category = label_map[predictions.item()]
    return "la categorie de votre article est: "+predicted_category

def LstmModel(data):
    # Prétraitement de la nouvelle phrase
    new_sequence = tokenizerLstm.texts_to_sequences([data])
    new_padded_sequence = pad_sequences(new_sequence, maxlen=100)

    # Faire la prédiction
    predictions =Lstm_Model.predict(new_padded_sequence)

    # Convertir les probabilités en catégories
    predicted_category = np.argmax(predictions, axis=1)

    # Récupérer le nom de la catégorie prédite
    category_labels = ['سياسة', 'اقتصاد', 'صحة']

    predicted_category_label = category_labels[predicted_category[0]]
    return "la categorie de votre article est: "+predicted_category_label

header = st.container()
#detect_categorieKnn("سبق للذكاء الاصطناعي أن أثبت قدرته في تحليل صور الأجهزة الطبية، والنجاح في اختبارات طلاب الطب. أما الآن، فحان دور أداة جديدة قائمة على الذكاء الاصطناعي لإثبات قدرتها على قراءة التقارير التي يعدها الأطباء والتنبؤ بدقة بمخاطر الوفاة ودخول المستشفى مجدداً والمضاعفات المحتملة الأخرى")
with header:
    st.title("Our HiKArabClassify System")
st.write('''
# Application For Classifying Arabic Articles
''')
col1, col2 = st.columns([1,2])
input = ''
model = col1.selectbox('Select a model', options=('select model','KNN','Naive-Bais','LSTM','ArBert'), index=0)
data=col2.text_input("Enter Your Article:")
if st.button('classifier'):
    if data:
        if model=='ArBert':
             result = Model_Bert(data)
             st.write(result)
        elif model=='Naive-Bais':
            result =NaiveBaise_Model(data)
            st.write(result)
        elif model=='LSTM':
            result = LstmModel(data)
            st.write(result)
    else:
        st.write('Enter Your Data!!!')

