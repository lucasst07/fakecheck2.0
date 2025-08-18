from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import networkx as nx
import math
import re
import nltk
from nltk.corpus import stopwords
import unicodedata
import json
from networkx.readwrite import json_graph
import os

# Configurações iniciais do NLTK
nltk.download('stopwords')
nltk.download('rslp')

# Inicializa a aplicação FastAPI
app = FastAPI()

# Configura CORS para permitir acesso de qualquer origem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variáveis globais para os recursos carregados
G = None
nlp = None


# Modelo Pydantic para entrada (não será usado diretamente mas mantido para documentação)
class NoticiaRequest(BaseModel):
    conteudo: str


# Carrega recursos na inicialização da API
@app.on_event("startup")
def carregar_recursos():
    global G, nlp

    # Carrega o grafo de conhecimento
    try:
        with open("grafo_json.json", "r", encoding='utf-8') as f:
            loaded_data = json.load(f)
        G = json_graph.node_link_graph(loaded_data)
        print(f"✅ Grafo carregado: {len(G.nodes)} nós, {len(G.edges)} arestas")
    except Exception as e:
        print(f"❌ Erro ao carregar grafo: {str(e)}")
        raise RuntimeError("Falha ao carregar grafo de conhecimento") from e

    # Carrega o modelo do spaCy
    try:
        nlp = spacy.load("pt_core_news_md")
        print("✅ Modelo spaCy carregado")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo spaCy: {str(e)}")
        raise RuntimeError("Falha ao carregar modelo de linguagem") from e


# Função de pré-processamento
def pre_processamento(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    texto = re.sub(r'[^a-z\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    tokens = texto.split()
    stop_words = set(stopwords.words('portuguese'))
    tokens = [palavra for palavra in tokens if palavra not in stop_words]
    return ' '.join(tokens)


# Função de extração de triplas (usa o modelo spaCy carregado globalmente)
def extract_triples(text):
    doc = nlp(text)
    triples = []
    for sent in doc.sents:
        subj, pred, obj = None, None, None
        for token in sent:
            if "subj" in token.dep_:
                subj = token.text
            if token.pos_ == "VERB":
                pred = token.lemma_
            if "obj" in token.dep_ and token.dep_ in ("dobj", "obj"):
                obj = token.text
        if subj and pred and obj:
            triples.append((subj, pred, obj))
    return triples


# Função de cálculo de valor de verdade
def calcular_valor_verdade(grafo, sujeito, objeto):
    if sujeito not in grafo or objeto not in grafo:
        return 0.0
    try:
        caminhos = nx.all_shortest_paths(grafo, sujeito, objeto)
        menor_custo = float('inf')
        for caminho in caminhos:
            custo = sum(math.log(grafo.degree(nó)) for nó in caminho[1:-1])
            if custo < menor_custo:
                menor_custo = custo
        return 1 / (1 + menor_custo) if menor_custo != float('inf') else 0.0
    except nx.NetworkXNoPath:
        return 0.0


# Função orquestradora principal
def verificar_noticia_completa(sentenca):
    triplas = extract_triples(pre_processamento(sentenca))
    if not triplas:
        return 0.0, "Nenhuma tripla válida encontrada."

    valores_tau = []
    for s, p, o in triplas:
        tau = calcular_valor_verdade(G, s, o)
        valores_tau.append(tau)

    tau_final = sum(valores_tau) / len(valores_tau) if valores_tau else 0.0
    limiar = 0.80
    classificacao = "verdadeira" if tau_final > limiar else "falsa"
    return tau_final, classificacao


# Endpoint principal da API
@app.post("/verificar-noticia")
async def verificar_noticia(noticia: str = Body(..., media_type="text/plain")):
    # Verifica se o texto tem pelo menos 50 caracteres
    if len(noticia.strip()) < 50:
        raise HTTPException(
            status_code=400,
            detail="Texto muito curto! Por favor, envie uma notícia com pelo menos 50 caracteres."
        )

    if G is None or nlp is None:
        raise HTTPException(
            status_code=500,
            detail="Recursos não foram carregados corretamente"
        )

    try:
        _, classificacao = verificar_noticia_completa(noticia)

        if classificacao == "falsa":
            return "Essa notícia pode ser falsa!😕 Recomendamos procurar informações em fontes confiáveis."
        else:
            return "Essa notícia parece ser verdadeira!😄 Recomendamos buscar fontes confiáveis antes de divulgá-la."

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro durante o processamento: {str(e)}"
        )

# Ponto de entrada para execução direta
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
