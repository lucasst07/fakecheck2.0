from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Union
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
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nltk.download('stopwords', quiet=True)
    nltk.download('rslp', quiet=True)
except Exception as e:
    logger.error(f"Erro ao baixar recursos NLTK: {e}")

app = FastAPI(
    title="FakeCheck API",
    description="API for fake news detection",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

G = None
nlp = None


class NoticiaRequest(BaseModel):
    conteudo: str


class VerificacaoResponse(BaseModel):
    resultado: str
    confianca: float
    classificacao: str


base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, "grafo_json.json")


@app.on_event("startup")
async def carregar_recursos():
    global G, nlp

    try:
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Arquivo {json_path} não encontrado")

        with open(json_path, "r", encoding='utf-8') as f:
            loaded_data = json.load(f)
        G = json_graph.node_link_graph(loaded_data)
        logger.info(f"✅ Loaded graph: {len(G.nodes)} nós, {len(G.edges)} arestas")
    except Exception as e:
        logger.error(f"❌ Erro ao load graph: {str(e)}")
        raise RuntimeError("Failed to load knowledge graph") from e
    try:
        nlp = spacy.load("pt_core_news_lg")
        logger.info("✅ spaCy model loaded")
    except OSError:
        logger.error("❌ Model pt_core_news_lg not found")
        logger.info("Execute: python -m spacy download pt_core_news_lg")
        raise RuntimeError("spaCy model not installed")
    except Exception as e:
        logger.error(f"❌ Erro ao load model spaCy: {str(e)}")
        raise RuntimeError("Failed to load language model") from e


def pre_processamento(texto):
    if not texto:
        return ""

    texto = texto.lower()
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    texto = re.sub(r'[^a-z\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()

    tokens = texto.split()
    try:
        stop_words = set(stopwords.words('portuguese'))
        tokens = [palavra for palavra in tokens if palavra not in stop_words and len(palavra) > 2]
    except Exception:
        logger.warning("Stopwords are not available, using text without filter")
    return ' '.join(tokens)


def extract_triples(text):
    if not text or nlp is None:
        return []

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


def calcular_valor_verdade(grafo, sujeito, objeto):
    if not grafo or sujeito not in grafo or objeto not in grafo:
        return 0.0

    try:
        caminhos = list(nx.all_shortest_paths(grafo, sujeito, objeto))
        if not caminhos:
            return 0.0

        menor_custo = float('inf')
        for caminho in caminhos:
            if len(caminho) <= 1:
                continue
            custo = sum(math.log(max(grafo.degree(nó), 1)) for nó in caminho[1:-1])
            menor_custo = min(menor_custo, custo)

        return 1 / (1 + menor_custo) if menor_custo != float('inf') else 0.0
    except nx.NetworkXNoPath:
        return 0.0
    except Exception as e:
        logger.error(f"Error not calculating truth value: {e}")
        return 0.0


def verificar_noticia_completa(sentenca):
    if not sentenca:
        return 0.0, "falsa"

    texto_processado = pre_processamento(sentenca)
    triplas = extract_triples(texto_processado)

    if not triplas:
        return 0.0, "falsa"

    valores_tau = []
    for s, p, o in triplas:
        tau = calcular_valor_verdade(G, s, o)
        valores_tau.append(tau)

    tau_final = sum(valores_tau) / len(valores_tau) if valores_tau else 0.0
    limiar = 0.80
    classificacao = "verdadeira" if tau_final > limiar else "falsa"

    return tau_final, classificacao


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "loaded_graph": G is not None,
        "loaded_nlp_model": nlp is not None
    }


@app.options("/verificar-noticia")
async def options_verificar_noticia():
    return {"message": "OK"}


import re

@app.post("/verificar-noticia", response_model=dict)
async def verificar_noticia(dados: Union[NoticiaRequest, str] = Body(...)):
    logger.info(f"Raw data received: {repr(dados)}")
    logger.info(f"Data type: {type(dados)}")

    try:
        if isinstance(dados, NoticiaRequest):
            noticia = dados.conteudo
            logger.info("Data received as JSON (NoticiaRequest)")
        elif isinstance(dados, str):
            if 'Content-Disposition: form-data' in dados:
                logger.info("Multipart form-data detected, extracting content...")
                match = re.search(r'Content-Disposition: form-data.*?\r\n\r\n(.*?)\r\n-+', dados, re.DOTALL)
                if match:
                    noticia = match.group(1).strip()
                    logger.info(f"Content extracted from form-data: '{noticia}'")
                else:
                    noticia = ""
                    logger.warning("Could not extract content from form-data")
            else:
                noticia = dados
                logger.info("Data received as plain string")
        else:
            noticia = str(dados) if dados is not None else ""
            logger.info(f"Data converted to string: {type(dados)}")
    except Exception as e:
        logger.error(f"Error while processing input data: {e}")
        raise HTTPException(
            status_code=400,
            detail="Invalid data format"
        )

    logger.info(f"Final processed news: '{repr(noticia)}'")
    logger.info(f"Final news size: {len(noticia)}")

    texto_limpo = noticia.strip() if noticia else ""

    if not texto_limpo or len(texto_limpo) < 50:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Text too short!",
                "message": "Please provide a news text with at least 50 characters.",
                "current_size": len(texto_limpo),
                "min_size": 50,
                "received_text": texto_limpo[:100] + "..." if len(texto_limpo) > 100 else texto_limpo
            }
        )

    if G is None or nlp is None:
        raise HTTPException(
            status_code=500,
            detail="Resources were not loaded correctly. Check server logs."
        )

    try:
        confianca, classificacao = verificar_noticia_completa(noticia)

        if classificacao == "falsa":
            resultado = "FALSE"
        else:
            resultado = "TRUE"

        return {
            "result": resultado,
            "text_size": len(noticia)
        }

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Check logs."
        )
