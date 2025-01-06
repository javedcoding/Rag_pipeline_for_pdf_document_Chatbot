# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 13:08:40 2024

@author: Mashnunul Huq
"""
#PDF handling related libraries
import gdown
import os
import pdfplumber
from langchain_community.document_loaders import PyPDFLoader
#Database Related Libraries
import weaviate
from weaviate.classes.init import Auth
from langchain_weaviate.vectorstores import WeaviateVectorStore
#RAG Retriever Related Libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.chains import RetrievalQA
#LLM Related Libraries
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import ( AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, )


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
            
def detect_selected_chapter_pages(starting_chapter_number=1, ending_chapter_number=5, chapter_keyword="CHAPTER"):
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

        print(f"from page {starting_page_number} to {ending_page_number}.")
    return starting_page_number, ending_page_number, downloaded_file_path

def load_data_in_chunks(chunk_size=200, chunk_overlap=30):
    '''
    '''
    #Get the pages for selected Chapters (1-5)
    startingPage, endingPage, downloaded_path = detect_selected_chapter_pages()
    #Load the pdf again for transformers loading requirements
    loader=PyPDFLoader(downloaded_path)
    documents = loader.load()
    #Select the Pages
    selected_pages = documents[startingPage : endingPage]
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(selected_pages)
    print("Chunks creation successful.")
    
    return chunks

def connect_weaviate_vector_store(WEAVIATE_CLUSTER_URL, WEAVIATE_API_KEY, HF_TOKEN, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    '''
    '''
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url = WEAVIATE_CLUSTER_URL,
            auth_credentials = Auth.api_key(WEAVIATE_API_KEY),
            headers={
                "X-HuggingFace-Api-Key": HF_TOKEN
            },
        )
        print("Vector Store connection successful.")
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        print("Vector Embedding model loading successful.")
        return client, embedding_model
    except Exception as e:
        print(f"Error Connecting to weaviate: {e}")

def upload_data_weaviate_vector_store(weaviate_cloud_client, embedding_model, index_name="HYBRIDRAG"):
    '''
    '''
    #Get the chunks first
    chunks = load_data_in_chunks()
    
    #For vector store use chunks to make a dense vector embeddings (without 0s)
    vector_store = WeaviateVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        client=weaviate_cloud_client,
        index_name=index_name,
    )
    print("Vector Store created.")
    
def load_retrievers_weaviate(weaviate_cloud_client, embedding_model, COHERE_API_KEY, index_name="HYBRIDRAG", text_key='text', cohere_model="rerank-english-v3.0", alpha=0.5, k=5):
    '''
    '''
    try:
        
        vector_store = WeaviateVectorStore(embedding= embedding_model, client=weaviate_cloud_client, index_name=index_name, text_key=text_key)
        try:
            retriever = vector_store.as_retriever(search_kwargs={"alpha": 0.5, "k": 5})
            print("Normal Retriever Created successfully")
        except Exception as e:
            print(f"Error Creating Retriever: {e}")
        compressor = CohereRerank(model=cohere_model, cohere_api_key=COHERE_API_KEY)
        try:
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever
            )
            print("Compression Retriever Created successfully")
        except Exception as e:
            print(f"Error Creating Compression Retriever: {e}")
        print("Retriever and Compressor Created successfully")
        return retriever, compression_retriever
    except Exception as e:
        print(f"Error Creating Retriever: {e}")



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
    print("tokenizer successfully loaded")
    
    #invoke the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    
    print("model and tokenizer loaded successfully.")
    
    return model, tokenizer 

def llm_pipeline(model, tokenizer, max_length=2048):
    
    #load the llm model and corresponding tokenizer for llm related word or sentence tockenizing
        
    main_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device_map="auto",
        max_length=max_length,
        do_sample=True,
        top_k=5,
        max_new_tokens=100,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    return main_pipeline

def rag_pipeline(model, tokenizer, retriever, chain_type="stuff"):
    '''

    '''
    pipeline= llm_pipeline(model, tokenizer)
    llm = HuggingFacePipeline(pipeline=pipeline)
    
    
    hybrid_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type=chain_type, 
        retriever=retriever
    )
    
    print("Pipeline Ready.")
    return hybrid_chain