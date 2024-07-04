import streamlit as st
import pandas as pd
import numpy as np
from langchain_community.chat_models import ChatOpenAI
from service.legal_nli import LegalNLI

def load_data():
    data = pd.read_csv('./data/train.csv')
    return data


st.title('Dataset')
st.subheader('Data Exploration')

data = load_data()
st.write(data)

st.subheader('Data distribution per label')
hist_values = np.histogram(data['label'], bins=2, range=(0,1))[0]
st.bar_chart(hist_values)


st.title("🦜🔗 Langchain Quickstart App")
