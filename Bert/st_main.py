import streamlit as st

from pages import BertModel  , LlammaModel , Rapport

st.set_page_config(layout="wide")
page_options = {
    "Bert model": BertModel,
    "LLamma model": LlammaModel,
    'Rapport':  Rapport
}
st.markdown(
        """
        <h1 style='text-align: center;'>Welcome to 'No Toxic Messages App'</h1>
        """,
        unsafe_allow_html=True  # Permet l'utilisation de HTML dans Streamlit
    )
with st.container(border=True):
    st.image('Bert/medias/no_hate.png')
st.subheader (':point_left: Naviguer à travers les pages de l\'application')
def page1():
    st.markdown('Bert model')





