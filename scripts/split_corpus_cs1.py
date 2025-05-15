import numpy as numpy
import os
#import pandas as pd
path_to_data = os.path.join(os.getcwd(),"data")
path_to_corpus = os.path.join(os.getcwd(),"data","corpus")
path_to_corpus_splitted = os.path.join(path_to_data,"corpus_splitted_cs1")

from generate_chunks import chunking_strategy_2,chunking_strategy_4


def split_debates(f,chunk_strat,l) : 
    print(f"Split and saving file :{f}")
    base_f = f.split(".")[0]
    os.makedirs(os.path.join(path_to_corpus_splitted,base_f),exist_ok=True)

    text = open(os.path.join(path_to_corpus,f)).read()
    blocs = chunk_strat(text,l)

    for i , b in enumerate(blocs) : 
        with open(os.path.join(path_to_corpus_splitted,base_f,f"{base_f}_{i:03}.txt"),"w") as file :
            file.write(b)

corpus_files = os.listdir(path_to_corpus)
for cf in sorted(corpus_files) : 
    split_debates(cf,chunking_strategy_2,10000)