import streamlit as st

DEFAULT_TEXT = "въведете текст тук"
def render_nli_form(form_name, exp=DEFAULT_TEXT, hyp=DEFAULT_TEXT, ans=DEFAULT_TEXT, callback:callable=None):
    with st.form(form_name):
        explaination = st.text_area("Увод (explaination):", exp)
        hypothesis  = st.text_area("Въпрос (question):", hyp)
        answer  = st.text_area("Потенциален отговор (answer):", ans)

        submitted = st.form_submit_button("Изпълни")
        if submitted:
            callback(explaination, hypothesis, answer)

def render_load_data_modal(data, callback):
    @st.experimental_dialog("Избери пример от корпуса", width="large")
    def select():
        st.write(data)
        input_idx = st.text_input("idx")

        if st.button("Зареди"):
            callback(input_idx)
            st.rerun()

    if st.button("Зареди пример от корпуса"):
        select()
