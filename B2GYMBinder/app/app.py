
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st

st.set_page_config(layout='wide', page_title='Home B2-GYM Assistant RAG', page_icon='🤖')

from qa import QAModel

@st.cache_resource()
def initialize_qa_model():
    return QAModel()

class QAApp:
    def __init__(self):
        self.qa_model = initialize_qa_model()

    def generate_answer(self):

        request = st.session_state.request
        qa_response = self.qa_model(request = request)

        if request is not None:
            if st.session_state.messages and request.strip() != "":
                response = qa_response
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    def run_app(self):
        st.title("Ask me anything about the B2-GYM 🤖👌")

        if st.button("Clear Chat"):
            st.session_state.messages = []

        if "messages" not in st.session_state:
                st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_input:= st.chat_input("Write a Question 💪", key="request"):
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            self.generate_answer()




if __name__ == '__main__':
    qa = QAApp()
    qa.run_app()