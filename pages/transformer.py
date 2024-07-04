import streamlit as st
import pandas as pd

from components.common import render_nli_form, DEFAULT_TEXT, render_load_data_modal
from service.summarization import Summarization
from service.legal_nli import LegalNLI

data = pd.read_csv('./data/train.csv')
legal_nli = LegalNLI()

if 'exp' not in st.session_state:
    st.session_state['exp'] = DEFAULT_TEXT
    st.session_state['que'] = DEFAULT_TEXT
    st.session_state['ans'] = DEFAULT_TEXT

def select_example(idx):
    st.session_state['exp'] = data.iloc[int(idx)]['explanation']
    st.session_state['que'] = data.iloc[int(idx)]['question']
    st.session_state['ans'] = data.iloc[int(idx)]['answer']

def callback(explaination, hypothesis, answer):
    response = legal_nli.predict(explaination+hypothesis, answer)
    st.write(response.to_json())

render_load_data_modal(data, select_example)

if st.button("Резюмирай увода с BART"):
    summarization_service = Summarization()
    st.session_state.exp = summarization_service.summarize_bart(st.session_state.exp)
    st.rerun()

if st.button("Резюмирай увода с ChatGPT-4"):
    summarization_service = Summarization()
    st.session_state.exp = summarization_service.summarize_gpt4(st.session_state.exp)
    st.rerun()

render_nli_form("nli_form", st.session_state.exp, st.session_state.que, st.session_state.ans, callback)
