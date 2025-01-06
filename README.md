# Rag_pipeline_for_pdf_document_Chatbot

# Awesome README [![Awesome](https://cdn.jsdelivr.net/gh/sindresorhus/awesome@d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome#readme)
> A curated list of awesome READMEs
> <img src="icon.png" align="right"/>

This is a repository for PDF reading and based on the pdf document a chatbot can answer questions. Here a Harrypotter Book is used from a google drive Download.

## How to turn on the ChatBot
Use the following command to install required libraries of Python in your environment.
```
pip install -r requirements. txt
```
Now we need to set our environment variables fro Credentials of Weaviate cluster url, weaviate api key, huggingface reading token and cohere api key. Giving a demo one below:
```
set WEAVIATE_CLUSTER_URL=https://46e.c0.europe-west3.gcp.weaviate.cloud 
set WEAVIATE_API_KEY=VLTCSNafVUY3V3MGKLQMPAj
set COHERE_API_KEY=lf9Nlrzb08B2RLXTx8IgK6kBXt
set HF_TOKEN=hf_WcywCSvBGmtMl
```

After that run the FastAPI based chatbot being in the folder where the main.py file is situated. A simple Javascript is used to make the workable Chatbot.
```
uvicorn main:app --reload
```
Here the --reload is optional to reload the application if changes are made.

## How the Application Works
You will the below page when you hit the localhost:8000/ https page:

![Chatbot ground page](https://github.com/user-attachments/assets/3ef390a3-9175-4f2e-a489-370c52acc013)

### Give the Question
Write the question related to your file in the below picture. It can run upto infinite question answering.
![How to write Multiple Question](https://github.com/user-attachments/assets/c1a6cbba-c5a5-4091-af8e-e38e8b84cc00)

### Check your machine is handling Application Correctly
This application is running on a local machine. My machine is quite strong to handle the Application (Asus Rog Strix G15, AMD 5900 Ryzen 9, RAM 64GB, Nvidia RTX3060)
![running on local machine](https://github.com/user-attachments/assets/6c2df858-6dfa-42ce-bf16-8eaa3897829b)


## How the Application works
At first the application downloads the file from the google drive. In the model.py file you may check this Function "extract_file(link, fileName, output_folder="downloads")" for your Downloadable file.
Later having the extracted Chapters pages [Function "detect_selected_chapter_pages"] the pdf will be loaded with PyPDFLoader of Langchain Community library.

### Models Used:
| Model Names | Using Purpose |
| ------------- | ------------- |
| BM25Retriever  | For Extracting Sparse Embeddings of Document Chunks Keyword Searching |
| BAAI/bge-large-en-v1.5  | For Extracting Dense Embeddings of Document Chunks  |
| HuggingFaceH4/zephyr-7b-beta  | For LLM final result  |

With the hybrid search EnsembleRetriever of Langchain dense_weight=0.6, sparse_weight=0.4 weights are given. Keyword search is less important than the context for this work that is why dense embeddings has slightly higher weights given. Then the Huggingface tockenizer is used for chunks tocknizing and the pipeline is below:

```
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
```

The final result is comming out of the below Hybrid Rag pipeline:
```
hybrid_chain=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=ensemble_retriever
        )
```
The file main.py holds the calling of the functions of model.py with some error handling for FastAPI frontend.
