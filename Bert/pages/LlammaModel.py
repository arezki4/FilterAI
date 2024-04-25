import streamlit as st 
import pandas as pd
st.title("Résultat de LLama guard modèl 🦙")


st.subheader("Testing : ")
st.video('Bert/medias/for_st_app.mp4')

st.subheader("Résultat : ")
resultat = pd.read_csv("Llama/resultat_llama.csv")
st.write(resultat)