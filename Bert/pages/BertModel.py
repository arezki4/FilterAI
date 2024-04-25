import streamlit as st
import torch
from pages.notebook_pytorch import tokenizer, ToxicCommentClassifier

st.subheader('Freindly Chat App 🥰 ')
#st.image('no_hate.png')

if 'messages' not in st.session_state:
    st.session_state.messages =[]

for message in st.session_state.messages:
    if message['note']== 'Message inapproprié' :
        avatar =avatar = '🟥'
    else :
        avatar = '🟩'
    with st.chat_message(message['role'],avatar= avatar):
        st.markdown(message['content'])
        st.markdown(f"{message['note']}: {message['prediction']}")

CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

model = ToxicCommentClassifier(n_classes=6)
model.load_state_dict(torch.load('Bert/modele_pl.pt', map_location=torch.device('cpu')))
model.eval()
prompt = st.chat_input('Saisir un message ..')
if prompt :
    with st.chat_message('user'):
        st.markdown(prompt)
    
    encoding = tokenizer.encode_plus(
    prompt,
    add_special_tokens=True,
    max_length=128,
    return_token_type_ids=False,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt"
    )
    model.eval()
    _, preds = model(encoding["input_ids"], encoding["attention_mask"])
    preds = preds.flatten().detach().numpy()
    predictions = []
    for idx, label in enumerate(CLASSES):
        if preds[idx] > 0.5:
            predictions.append((label, round(preds[idx]*100, 2)))
    if len(predictions )>0:
        note = 'Message inapproprié' 
        
    else: 
        note = 'Message clean'
        
    st.session_state.messages.append({'role':'user', 'content': prompt, 'note':note,'prediction':f'{ predictions}'})
    if len(predictions)>0:
        st.write('Votre message ne peut pas être envoyé :red_circle: :warning: :heavy_exclamation_mark:')
        for label, score in predictions:
                st.write(f"- '{label}' : {score} %")
    else:
        st.subheader("Votre message est clean :white_check_mark:")
    
  
    

