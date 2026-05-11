import re, os, json
from time import time
from http import client
from datetime import datetime
from crewai_tools import SerperDevTool
from crewai_tools import BaseTool
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import OnlinePDFLoader, WebBaseLoader, CSVLoader, EverNoteLoader, PyMuPDFLoader, TextLoader, UnstructuredEmailLoader, UnstructuredEPubLoader, UnstructuredHTMLLoader,UnstructuredMarkdownLoader,UnstructuredODTLoader,UnstructuredPowerPointLoader,UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from tqdm.autonotebook import tqdm, trange
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

class DataExtractor:
    all_splits = []
    invalid_urls = []
    count = 1
    failed_count = 0
    n_path = ''
    persist_directory = os.environ.get('PERSIST_DIRECTORY', 'systematic_review_sdg_db')
    model_name_best = os.environ.get("MODEL", "llama-3.1-70b-versatile")
    embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME', 'all-MiniLM-L6-v2')
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    
    def __init__(self):
        self.all_splits = []
        self.invalid_urls = []
        self.count = 1
        self.failed_count = 0
        self.n_path = 'C:\\Users\\MandlenkosiNgwenya\\Documents\\experiments\\output\\'
        print('Data extract initialized')

    @tool("Tool extracts data from URLs and store in chromadb")
    def store_in_vector_db(self):
        """This tool is used for extracting text from URL and save it in the Chromadb for full-text search, document storage, vector search and metadata filtering."""

        xlink=''
        start_time = time.time()
        
        # Data retrieved must be saved when the data collector agent runs
        try:
            with open(self.n_path + 'data_retrieved.json', 'r') as f:
                new_retrieved_data = json.load(f)
        except Exception as ex:
            print(f"ERROR HAPPENED WHILE READING FILE {self.n_path} data_retrieved_sdgs.json. Original Exception: {ex}")
        
        for dataItem in new_retrieved_data[:20]:
            query =  dataItem['query']
            title = dataItem['title']
            xlink = dataItem['link']
            print("**************************************************************************************************************************************************************************")
            try:
                if (str(xlink).endswith(".pdf")):
                    print("Online PDF")
                    loader = PyMuPDFLoader(xlink)
                else:
                    loader = WebBaseLoader(xlink)
                    
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)
                self.all_splits.extend(splits)
                # vectorstore = Chroma.from_documents(all_splits, embedding=embeddings, persist_directory=PERSIST_DIRECTORY)
                count +=1
                # print(f"Split into {len(splits)} chunks of text (max. {chunk_size} tokens each)")
                print(f'Processed >>> {count}', ":::",title, "---", xlink)
            except Exception as ex:
                print(f"Error happened at {xlink}. Origianl Exception: {ex}")
                self.invalid_urls.append(xlink)
                self.failed_count += 1
                print(f"Failed items >>> {self.failed_count}")
                

                continue
            
        print("===========================================================================================================================================================================")
        end_time = time()
        diff = end_time - start_time
        print(f'Time taken in seconds {diff}')  
        print(f'Failed Items {self.failed_count}')  
        print(f'Succeeded Items {self.count}')  
    
        print(f"Appending to existing vectorstore at {self.persist_directory}")
        start_time = time()
        Chroma.from_documents(self.all_splits, embedding=self.embeddings, persist_directory=self.persist_directory)
        end_time = time()
        save_diff = end_time - start_time
        print(f'Time taken in seconds to save in ChromaDb {save_diff}') 
        return self.all_splits, self.invalid_urls