import pandas as pd
import numpy
import os
from sentence_transformers import SentenceTransformer

import chromadb
from chromadb.utils import embedding_functions

BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR,'data')
path_to_embeddings = os.path.join(DATA_DIR,"corpus_splitted_cs1")

client = chromadb.PersistentClient(path = os.path.join(DATA_DIR,"embeddings_cs1"))
st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="Alibaba-NLP/gte-multilingual-base",trust_remote_code=True,device="cuda",normalize_embeddings=True)

seances = sorted(os.listdir(path_to_embeddings))


for i , s in enumerate(seances) : 
    print(f"Embedding de la séance numéro {i} : {s}")
    files = sorted(os.listdir(os.path.join(path_to_embeddings,s)))
    docs_seance = []
    for f in files :
        text = open(os.path.join(path_to_embeddings,s,f)).read() 
        docs_seance.append(text)
    collection = client.create_collection(name=s,embedding_function=st_ef,metadata={"hnsw:space":"l2"})
    collection.add(documents=docs_seance,ids=files)
    print("--------------------")