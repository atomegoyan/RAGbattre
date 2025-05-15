import json
import os
import numpy as np
import pandas as pd
import regex as re

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunking_strategy_0(text,chunk_size) :
    text_splitter_1 = RecursiveCharacterTextSplitter(
        separators=["\nM\.","\n{2,}","\n"],
        chunk_size=chunk_size,
        chunk_overlap=1,
        length_function=len,
        is_separator_regex=True,
        )
    blocs = text_splitter_1.create_documents([text])
    return [b.page_content for b in blocs]

def chunking_strategy_1(text) : 
    text_splitter_1 = RecursiveCharacterTextSplitter(separators=['\n+[a-z]{0,2}[A-ZÉÈÀÊÔ \-\'0-9\—\.]{8,}?.*\n',"[0-9\-\—\. ]{3,}[a-z\&\'\&]*[A-ZÉÈÊÀÔ\-\.\—\°\:\; ']{10,}.*\n*"],
        chunk_size=1,
        chunk_overlap=1,
        length_function=len,
        is_separator_regex=True,
        )
    blocs = text_splitter_1.create_documents([text])
    return [b.page_content for b in blocs]

def chunking_strategy_2(text,chunk_size) :
    
    text_splitter_1 = RecursiveCharacterTextSplitter(separators=['\n+[a-z]{0,2}[A-ZÉÈÀÊÔ \-\'0-9\—\.]{8,}?.*\n'],
        chunk_size=1,
        chunk_overlap=1,
        length_function=len,
        is_separator_regex=True,
        )
    
    text_splitter_2 = RecursiveCharacterTextSplitter(
        separators=["[0-9\-\—\. ]{3,}[a-z\&\'\&]*[A-ZÉÈÊÀÔ\-\.\—\°\:\; ']{10,}.*\n*","\nM\.","\n{2,}","\n"],
        chunk_size=chunk_size,
        chunk_overlap=1,
        length_function=len,
        is_separator_regex=True,
        )

    docs = text_splitter_1.create_documents([text])
    docs = text_splitter_2.create_documents([d.page_content for d in docs])
    return [d.page_content for d in docs]



def chunking_strategy_3(text,chunk_size) :
    
    text_splitter_1 = RecursiveCharacterTextSplitter(separators=['\n+[a-z]{0,2}[A-ZÉÈÀÊÔ \-\'0-9\—\.]{8,}?.*\n'],
        chunk_size=1,
        chunk_overlap=1,
        length_function=len,
        is_separator_regex=True,
        )
    
    text_splitter_2 = RecursiveCharacterTextSplitter(
        separators=["[0-9\-\—\. ]{3,}[a-z\&\'\&]*[A-ZÉÈÊÀÔ\-\.\—\°\:\; ']{10,}.*\n*"],
        chunk_size=chunk_size,
        chunk_overlap=1,
        length_function=len,
        is_separator_regex=True,
        )

    docs = text_splitter_1.create_documents([text])
    docs = text_splitter_2.create_documents([d.page_content for d in docs])
    return [d.page_content for d in docs]


def chunking_strategy_4(text,chunk_size=1000) :
    
    text_splitter_1 = RecursiveCharacterTextSplitter(separators=['\n+[a-z]{0,2}[A-ZÉÈÀÊÔ \-\'0-9\—\.]{8,}?.*\n'],
        chunk_size=1,
        chunk_overlap=1,
        length_function=len,
        is_separator_regex=True,
        )
    
    text_splitter_2 = RecursiveCharacterTextSplitter(
        separators=["[0-9\-\—\. ]{3,}[a-z\&\'\&]*[A-ZÉÈÊÀÔ\-\.\—\°\:\; ']{10,}.*\n*","\nM\.","\n{2,}","\n"],
        chunk_size=chunk_size,
        chunk_overlap=1,
        length_function=len,
        is_separator_regex=True,
        )

    docs = text_splitter_1.create_documents([text])
    docs = text_splitter_2.create_documents([d.page_content for d in docs])
    return [d.page_content for d in docs]

def chunking_strategy_pp(text) :
    
    text_splitter_1 = RecursiveCharacterTextSplitter(separators=['\n+[a-z]{0,2}[A-ZÉÈÀÊÔ \-\'0-9\—\.]{8,}?.*\n'],
        chunk_size=1,
        chunk_overlap=1,
        length_function=len,
        is_separator_regex=True,
        )
    
    text_splitter_2 = RecursiveCharacterTextSplitter(
        separators=["[0-9\-\—\. ]{3,}[a-z\&\'\&]*[A-ZÉÈÊÀÔ\-\.\—\°\:\; ']{10,}.*\n*","\nM\."],
        chunk_size=1,
        chunk_overlap=1,
        length_function=len,
        is_separator_regex=True,
        )

    docs = text_splitter_1.create_documents([text])
    docs = text_splitter_2.create_documents([d.page_content for d in docs])
    return [d.page_content for d in docs]