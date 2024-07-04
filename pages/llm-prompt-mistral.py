import streamlit as st
import pandas as pd
from service.legal_nli import LegalNLI
from components.common import render_nli_form, DEFAULT_TEXT, render_load_data_modal

data = pd.read_csv('./data/train.csv')

if 'exp' not in st.session_state:
    st.session_state['exp'] = DEFAULT_TEXT
    st.session_state['que'] = DEFAULT_TEXT
    st.session_state['ans'] = DEFAULT_TEXT

def select_example(idx):
    st.session_state['exp'] = data.iloc[int(idx)]['explanation']
    st.session_state['que'] = data.iloc[int(idx)]['question']
    st.session_state['ans'] = data.iloc[int(idx)]['answer']

def callback(explaination, hypothesis, answer):
    legalNLI = LegalNLI()
    response = legalNLI.predict_mistral(explaination, hypothesis, answer)
    st.info(response)

## Page
render_load_data_modal(data, select_example)
render_nli_form("nli_form", st.session_state.exp, st.session_state.que, st.session_state.ans, callback)
