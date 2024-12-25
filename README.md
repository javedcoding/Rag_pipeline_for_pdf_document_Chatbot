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
At first the application downloads the file from the google drive
