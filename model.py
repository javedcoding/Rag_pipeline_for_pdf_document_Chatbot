# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 13:08:40 2024

@author: mashn
"""

import gdown
import os
import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, )
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA

def extract_file(link, fileName, output_folder="downloads"):
    '''
    This is a function to automatically download a file from the Google drive Folders
    Caution: if the file is not inside a folder need this to be changed

    Parameters:
        link : This is the url of the shared google file
        fileName : The fileName that has to be found
        output_folder(optional) : If you want downloaded files elsewhere only then give full path

    Returns:
        path : the found file path
        or
        downloaded_files : all the file paths inside the desired output Folder
    '''
    os.makedirs(output_folder, exist_ok=True)
    gdown.download_folder(link, output=output_folder)
    downloaded_files = []
    for root, _, files in os.walk(output_folder):
        for file in files:
            full_path = os.path.join(root, file)
            downloaded_files.append(full_path)
        for path in downloaded_files:
            if os.path.isfile(path) and os.path.basename(path).lower() == fileName.lower():
                return path
            else:
                return downloaded_files 
            
def detect_selected_chapter_pages(starting_chapter_number, ending_chapter_number, chapter_keyword="CHAPTER"):
    '''
    This Function finds out the First page of the first chapter and last page of the last chapter

    Parameters:
        pdf_path(string) : Absolute or Relevant path of the pdf file
        starting_chapter_number(int) : Starting chapter number 
        ending_chapter_number(int) : Last chapter for selection
        chapter_keyword(string) : default is CHAPTER but if you want to find it with other keywords then use
    '''
    #Download the file at first
    downloaded_file_path = extract_file(
        link="https://drive.google.com/drive/folders/1YbkmtTprj86-uFs0et7f0sJIXHcMATJR?usp=sharing", 
        fileName="Harry Potter And The Deathly Hallows.pdf")
    with pdfplumber.open(downloaded_file_path) as pdf:
        starting_page_number = None
        ending_page_number = None
        
        # Iterate through each page in the PDF
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            
            # Check if the chapter title contains the keyword
            if text and chapter_keyword in text.upper():
                
                if starting_page_number is None and f"{chapter_keyword} {starting_chapter_number}" in text.upper():
                    starting_page_number = i + 1 # Return the first page (1-based index)

                if f"{chapter_keyword} {ending_chapter_number+1}" in text.upper():
                    ending_page_number = i # Return the before page of the next chapter starting page
                    break
        if ending_page_number is None:
            ending_page_number = len(pdf.pages)
    return starting_page_number, ending_page_number, downloaded_file_path

def load_data_file_in_chunks(chunk_size=200, chunk_overlap=30):
    '''
    '''
    #Get the pages for selected Chapters (1-5)
    startinPpage, endingPage, downloaded_path = detect_selected_chapter_pages(1, 5)
    #Load the pdf again for transformers loading requirements
    loader=PyPDFLoader(downloaded_path)
    documents = loader.load()
    #Select the Pages
    selected_pages = documents[startinPpage : endingPage]
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(selected_pages)
    print("Chunks creation successful.")
    
    return chunks

def create_vector_store(HF_TOKEN, model_name="BAAI/bge-large-en-v1.5", dense_weight=0.6, sparse_weight=0.4):
    '''
    '''
    #Get the chunks first
    chunks = load_data_file_in_chunks()
    
    #For vector store use chunks to make a dense vector embeddings (without 0s)
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN, model_name=model_name)
    vectorStore = Chroma.from_documents(chunks, embeddings)
    vectorStore_retriever = vectorStore.as_retriever(search_kwargs={"k":3})
    
    #For the keyword searching use chunks to make sparse vector embeddings (with 0s for only matching keywords)
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 3
    
    #now combine these retrievers
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorStore_retriever, keyword_retriever], 
        weights=[dense_weight, sparse_weight]
        )
    
    print("Vector Store created.")
    
    return ensemble_retriever

def load_quantized_llm_model(model_name="HuggingFaceH4/zephyr-7b-beta"):
    '''
    Parameter:
        model_name: Name or Path of the model to be loaded.
    Return:
        model: loaded quantized configured model
    '''
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    #invoke the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, return_token_type_ids=False)
    tokenizer.bos_token_id = 1
    print("tockenizer successfully loaded")
    
    #invoke the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    
    print("model and tockenizer loaded successfully.")
    
    return model, tokenizer 

def llm_pipeline(model, tokenizer):
    ensemble_retriever= create_vector_store(HF_TOKEN="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")
    
    #load the llm model and corresponding tockenizer for llm related word or sentence tockenizing
        
    main_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=2048,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        )
    
    return main_pipeline, ensemble_retriever

def rag_pipeline(model, tockenizer):
    '''
    '''
    #Get the vectore store 
    
    pipeline, ensemble_retriever = llm_pipeline(model, tockenizer)
    llm = HuggingFacePipeline(pipeline=pipeline)
    
    
    hybrid_chain=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=ensemble_retriever
        )
    
    print("Pipeline Ready.")
    return hybrid_chain