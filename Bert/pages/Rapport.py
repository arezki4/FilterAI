import streamlit as st 

st.title('Rapport de projet ')

st.subheader('Context et objectif: ')
with st.container(border=True):
    st.write("L'objectif principal de notre projet est d'améliorer la sécurité et le bien-être sur Internet en proposant une solution de modération et de filtrage des messages. Aujourd'hui, de nombreux individus subissent des dommages mentaux en raison de propos irrespectueux ou offensants sur les plateformes en ligne."
             
             "Avec l'accessibilité massive à Internet, il est facile pour certains de publier des contenus nuisibles sans conscience des conséquences. Notre but est de développer un modèle capable de comprendre le contexte des messages afin d'appliquer des restrictions appropriées avant leur publication. L'objectif n'est pas de restreindre les conversations légitimes, mais plutôt de promouvoir un environnement en ligne plus sûr en identifiant et en bloquant les messages potentiellement nuisibles tout en préservant la liberté d'expression de manière équilibrée et responsable.")
    
st.subheader("État de l'art :")
with st.container(border= True):
    st.write("dans le domaine de la modération de contenu sur Internet comprend déjà des modèles basés sur des réseaux neuronaux récurrents (RNN) qui effectuent la classification de manière efficace. Cependant, notre approche se distingue par l'utilisation de techniques novatrices basées sur l'intelligence artificielle générative.")
    st.write("Notre objectif était de créer un modèle en utilisant des outils et des modèles basés sur l'ia générative pour la modération et le filtrage des messages. Contrairement aux approches conventionnelles de fine-tuning de modèles existants, notre approche impliquait une modification structurelle. Au lieu de simplement fine-tuner un modèle pré-entraîné, nous avons exploré la technique du 'model ensembling' qui repose sur le principe de la sagesse de la foule.")
    st.write("Plus précisément, nous avons combiné un modèle d'intelligence artificielle générative avec un réseau de neurones pour exploiter les avantages de chacun. Le modèle génératif a été utilisé pour extraire des caractéristiques sous forme d'embeddings (vecteurs pondérés représentant le texte) permettant une meilleure compréhension du contexte par rapport aux RNN traditionnels. Ensuite, nous avons gelé le processus du modèle génératif pour utiliser ces embeddings comme entrée pour un réseau de neurones chargé de la classification.")
    st.write("Cette approche  visait à améliorer la capacité de compréhension du contexte du modèle tout en tirant parti des avantages de plusieurs modèles combinés, aboutissant ainsi à une meilleure performance dans la modération et le filtrage des contenus en ligne.")
    
st.subheader('Dataset et modèls utilisés :')
with st.container(border = True):
    st.markdown('- Pour les données:')
    st.write("Un dataset de taille 159571 ligne et 8 colonnes contenant : id , comment_text , et les 6 classes : 'toxic','severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'. La distribution non équilibré des classes est comme suite :")
    st.image('Bert/medias/classes.png')
with st.container(border = True):
    st.markdown('Pour le model Bert : ')
    st.write("Notre modèle utilise BERT (Bidirectional Encoder Representations from Transformers), un modèle pré-entraîné réputé pour sa capacité à comprendre le contexte du texte. Ce modèle comprend une couche BERT, tirant parti de ses représentations denses de texte pour capturer les nuances et le sens des commentaires. Une couche de classification linéaire est ajoutée au-dessus de BERT pour prédire les probabilités associées à différentes classes toxiques")
    st.write("Initialement,BERT est utilisé pour encoder des entrés tokenisés et générer des représentations contextuelles. Ensuite, la sortie de BERT est alimentée dans la couche de classification linéaire, où une fonction d'activation sigmoid est appliquée pour obtenir des probabilités de classe. La fonction de perte utilisée pour entraîner le modèle est la BCELoss (Binary Cross-Entropy Loss), adaptée à la classification binaire des classes toxiques.")
    st.write("Le modèle est entraîné, validé et testé pour chaque étape du processus d'apprentissage. Pendant l'entraînement, l'optimiseur AdamW est utilisé pour mettre à jour les poids du modèle, et un scheduler est configuré pour ajuster le taux d'apprentissage au fil du temps.")
    st.image('Bert/medias/architecture.png')
    st.write('versions: ')
    st.write('- Transformers : 4.40.0')
    st.write('- Pytorch_lightning : 2.2.3' )
    st.write("- Torch : 2.2.1+cu121")
    st.write('- BertTokenizer')
with st.container(border = True):
     st.markdown('Pour Le model LLAMA : ') 
     st.write("Le deuxième modèle que nous avons utilisé est le modèle LLAMA guard de Meta, qui possède un impressionnant nombre de 8 milliards de paramètres. Ce modèle d'ia générative est extrêmement puissant et a été entraîné sur un corpus massif de 1,4 trillion de tokens, ce qui lui confère une compréhension approfondie du langage naturel.")
     st.write("Nous avons exploité ce modèle pour effectuer une tâche de classification multilabel avec 11 catégories différentes. Pour adapter le modèle à notre tâche spécifique, nous avons réalisé un processus de fine-tuning en utilisant une technique appelée 'prompt engineering' qui consiste à concevoir des prompts (instructions) spécifiques qui guident le modèle dans sa capacité à réaliser la classification multilabel des textes/messages.")
     st.image("Bert/medias/lammagaurd.png")
     st.write('versions: ')
     st.write('- Transformers : 4.40.0')
     st.write("- Torch : 2.2.1+cu121")
     st.write('- AutoTokenizer')
     st.write("- AutoModelForCausalLM")
st.subheader('Autre solutions envisagées :')
with st.container(border = True):
    st.write("- Fine Tuning de Bert avec pytorch sur un dataset personnalisé (l'ajout d'une nouvelle classe 'spam'): Performance faible")
    st.write('- DistilBertForSequenceClassification : Performance faible apres le fine tuning')
    st.write('- Fine tuning de Bert avec tensorflow : Problème avec la librairie tensorflow')  
    st.write('- Un modèle basé sur les embedings : Performance faible ')   
    st.write("- Gpt : Limitation d'utilisation gratuite")
    st.write('- LLama 2 : Performance faible ')   
     
    
st.subheader('Difficulté rencontré :')
with st.container(border = True):
    
    st.write("- Tensorflow:")
    st.write("Sur windows à partir de la version 2.10.0  ne supporte plus la gpu.")
    st.write("La version de tensorflow 2.10.0 n’est plus référencé sur pip avec l’environnement python nous avons dû nous rabattre sur conda qui ne détecte pas la gpu a cause du problème de variable d’environnement.")
    st.write("Sur debian 12 on a rencontré un problème de permission non accordé avec l’env python et un problème de variable globale avec conda pour spécifier le chemin de cuda.")
    st.write("Sur la WSL on a eu un problème de variable d’environnement qu’on a réglé grâce a stack overflow.")
    st.write("Les résultats de l'entraînement n'étaient pas du tout concluants que ce soit avec le dataset avec les spams ou non.")
    
    st.write("- Pytorch: ")
    st.write("Problème de dépendance avec cuda.")
    st.write("Problème d'absence de barre de progression qui faisait accumulée les logs dans le notebook et qui saturait la RAM ce qui conduisait au plantage de l'entraînement.")
    st.write("Problème avec le notebook sur vscode qui ne supporte pas les gros traitements effectués lors de l'entraînement du modèle.")
    st.write("Impossible d'entraîner le modèle sur tout le data set car trop volumineux pour pytorch, nous avons donc dû utiliser un échantillon.")

    st.write('- Meta-Llama:')
    st.write("Problème de mémoire graphique avec cuda, nos gpu ne sont pas assez performant, même problème sur le serveur quand on essaye de l'intégrer à streamlit.")
    st.write("Le modèle marche avec la cpu cela dit les réponses prennent trop de temps.")
    st.write("Fine tuning impossible à cause des problèmes de mémoire GPU.")

    st.write('- Dataset:')
    st.write("Nous avons dû faire l'entraînement sur un dataset tres inegalement repartie avec des labels très minoritaires par rapport a d’autre.")
    st.write("Nombre de labels très restreints avec un total de 6 labels.")

    st.write('- Serveur:')
    st.write("Etant donné que le serveur se trouve dans un réseau privée il était impossible de se connecter à des applications web et de partager des fichiers facilement pour cela on a décidé d’utiliser une solution de tunneling ‘ngrok’  combiné à un jupyter notebook afin de faciliter la navigation dans le projet et le transfert et le téléchargement de fichier.")

 
