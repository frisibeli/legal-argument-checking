import streamlit as st
import pandas as pd

from components.common import render_nli_form, DEFAULT_TEXT, render_load_data_modal
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

def callback(explanation, question, answer):
    windows = []
    window_size = 50
    window_step = 100

    cache_str = str(explanation) + " | " + str(question)
    cache = cache_str.split()  # Split on whitespace
    if len(cache) <= window_size:
        windows.append((cache_str, answer))
    else:
        while len(cache) > window_size:
            sentence = " ".join(cache[: window_step + window_size])
            windows.append((sentence, answer))
            cache = cache[window_step:]

    st.write(windows)

    responses = []
    for window in windows:
        response = legal_nli.predict(window[0], window[1])
        responses.append(response)
        st.info(response.to_json())


render_load_data_modal(data, select_example)
render_nli_form("nli_form", st.session_state.exp, st.session_state.que, st.session_state.ans, callback)
