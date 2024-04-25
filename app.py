import streamlit as st
import torch
#from hate_detection import  trained_model
from hate_detection import tokenizer
from hate_detection import ToxicCommentClassifier

st.markdown(
    """
    <h1 style='text-align: center;'>Welcome to 'No Toxic Messages App'</h1>
    """,
    unsafe_allow_html=True  # Permet l'utilisation de HTML dans Streamlit
)

st.image('no_hate.png')
test_example = st.text_input('Saisir un message à envoyer :point_down:', 'Text...')
envoyer = st.button('Envoyer le message')

model = ToxicCommentClassifier(n_classes=6)
model.load_state_dict(torch.load('modele2_pl.pt', map_location=torch.device('cpu')))
model.freeze()
CLASSES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


encoding = tokenizer.encode_plus(
    test_example,
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

#predictions

if envoyer: 
    if len(predictions) > 0:
        st.subheader("Votre message ne peut pas être envoyé :red_circle: :warning: :heavy_exclamation_mark:")

        st.write('Votre message est : ')
        for label, score in predictions:
            st.write(f"- '{label}' : {score} %")
    else:
        st.subheader("Votre message est clean :white_check_mark:")
        st.write(test_example)
        