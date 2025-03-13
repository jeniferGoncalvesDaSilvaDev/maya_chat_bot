import streamlit as st
from transformers import pipeline
import random
import nltk
from sentence_transformers import SentenceTransformer, util

# Baixar pacotes necessários para tokenização NLTK
nltk.download('punkt')

# Carregar o modelo de análise de sentimentos
sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Carregar o modelo de embeddings para análise semântica
modelo_embeddings = SentenceTransformer('all-MiniLM-L6-v2')

# Histórico de interações
historico_interacao = []

# Respostas baseadas no sentimento
respostas_positivas = [
    "Fico muito feliz em saber disso! Quer me contar mais?",
    "Isso é ótimo! O que mais te animou hoje?",
    "Adoro ouvir boas notícias! O que mais você gostaria de compartilhar?"
]

respostas_negativas = [
    "Sinto muito que esteja passando por isso. Quer conversar mais sobre o que está te incomodando?",
    "Lamento saber disso. Posso te ajudar de alguma forma?",
    "Estou aqui para ouvir. Me conta mais sobre o que está acontecendo."
]

respostas_neutras = [
    "Entendo. Pode me explicar um pouco mais?",
    "Interessante! Como você se sente em relação a isso?",
    "Isso faz sentido. Pode me dar mais detalhes?"
]

# Função para analisar o sentimento e responder
def analisar_sentimento_responder(texto_usuario):
    analise = sentiment_analyzer(texto_usuario)[0]
    sentimento = analise['label']
    score = analise['score']

    if sentimento == "LABEL_2":
        sentimento = "POSITIVE"
    elif sentimento == "LABEL_1":
        sentimento = "NEUTRAL"
    else:
        sentimento = "NEGATIVE"

    # Adicionar ao histórico de interações
    historico_interacao.append({"texto": texto_usuario, "sentimento": sentimento, "score": score})

    # Escolher resposta apropriada
    if sentimento == "POSITIVE":
        resposta = random.choice(respostas_positivas)
    elif sentimento == "NEGATIVE":
        resposta = random.choice(respostas_negativas)
    else:
        resposta = random.choice(respostas_neutras)

    return resposta, sentimento, score

# Função para encontrar respostas semelhantes na memória
def encontrar_resposta_semelhante(nova_entrada):
    if not historico_interacao:
        return None

    # Criar embedding da nova entrada
    embedding_nova_entrada = modelo_embeddings.encode(nova_entrada, convert_to_tensor=True)

    # Criar embeddings para interações anteriores
    embeddings_historico = [modelo_embeddings.encode(entry["texto"], convert_to_tensor=True) for entry in historico_interacao]

    # Calcular similaridade
    similaridades = [util.pytorch_cos_sim(embedding_nova_entrada, emb)[0][0].item() for emb in embeddings_historico]

    # Encontrar a interação mais semelhante
    indice_mais_similar = similaridades.index(max(similaridades))

    if similaridades[indice_mais_similar] > 0.7:
        return historico_interacao[indice_mais_similar]["texto"]

    return None

# Título da aplicação Streamlit
st.title("Assistente de Sentimentos - Mayá")

# Botão de "Sair"
if st.button("Sair"):
    st.write("Obrigado por interagir! Até mais!")
    st.stop()  # Interrompe a execução da interação

# Loop de interação com o usuário
texto_usuario = st.text_input("Como você está se sentindo hoje?", key="user_input")

if texto_usuario:
    # Verifica se já houve uma conversa semelhante antes
    resposta_semelhante = encontrar_resposta_semelhante(texto_usuario)
    if resposta_semelhante:
        st.write(f"Você já mencionou algo parecido antes: '{resposta_semelhante}'. Quer me contar mais sobre isso?")
    else:
        resposta, sentimento, score = analisar_sentimento_responder(texto_usuario)
        st.write(f"Mayá ({sentimento}, confiança: {score:.2f}): {resposta}")

    # Mostrar o histórico de interações
    if len(historico_interacao) > 10:
        st.subheader("Histórico de Interações:")
        for entry in historico_interacao:
            st.write(f"Texto: {entry['texto']} - Sentimento: {entry['sentimento']} (Confiança: {entry['score']:.2f})")
