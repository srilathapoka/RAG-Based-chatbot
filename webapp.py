import os
import google.generativeai as genai
from langchain.vectorstores import FAISS #this will be the vector database
from langchain_community.embeddings import HuggingFaceEmbeddings #to perform word embedding
from langchain.text_splitter import  RecursiveCharacterTextSplitter #this for chunking
from pypdf import PdfReader
import faiss
import streamlit as st
from pdfextractor import text_extractor_pdf


#create the main page
st.title(':green[RAG Based CHATBOT]')
tips='''Follow the steps to use this application
* Upload your PDF Document is sidebar
* write your query and start chatting with the CHATBOT
'''
st.subheader(tips)


#load pdf in sidebar
st.sidebar.title(':orange[UPLOAD YOUR DOCUMENT HERE (PDF only)]')
file_uploaded=st.sidebar.file_uploader('Upload file')
if file_uploaded:
    file_text=text_extractor_pdf(file_uploaded)
    #configure LLM
    key=os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=key)
    llm_model=genai.GenerativeModel('gemini-2.5-flash-lite')

    #configure embedding model
    embedding_model=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    #step 2:chunking (create chunks)
    splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
    chunks=splitter.split_text(file_text)

    #step 3: create FAISS vector store
    vector_store=FAISS.from_texts(chunks,embedding_model)

    #step 4:configure retriever
    retriever=vector_store.as_retriever(search_kwargs={'k':3})

    #lets create a function that takes queryand return the generated text
    def generate_response(query):
        #step 6:Retrieval (R)
        retrieved_docs=retriever.get_relevant_documents(query=query)
        context=' '.join([doc.page_content for doc in retrieved_docs])

        #step 7:write a Argumented prompt(A)
        prompt=f'''You are a helpful assitant using RAG
        Here is the context {context}
        the query asked by user is as follows={query}'''

        #step 9:Generation(G)
        content=llm_model.generate_content(prompt)
        return content.text


    #step 5:take the query
    
    #lets create a chatbot in order to start the conversation
    #start chat history
    #initialize chat if there is no history
    if 'history' not in st.session_state:
        st.session_state.history=[]

    #display the history
    for msg in st.session_state.history:
        if msg['role']=='user':
            st.write(f':green[User:] :blue[{msg['text']}]') 
        else:
            st.write(f':orange[Chatbot:] {msg['text']}')

    #input from the user(using streamlit form)
    with st.form('Chat Form ',clear_on_submit=True):
        user_input=st.text_input('Enter Your Text Here:')
        send=st.form_submit_button('Send')



    #start the conversation and append the output and query in history
    if user_input and send:

        st.session_state.history.append({'role':'user','text':user_input})

        model_output=generate_response(user_input)

        st.session_state.history.append({'role':'chatbot','text':model_output})

        st.rerun()






