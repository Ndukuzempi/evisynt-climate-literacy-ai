import re, os, json, csv
import stat
from time import time
from http import client
from datetime import datetime
from typing import Optional
from argon2 import Type
from crewai_tools import SerperDevTool, RagTool
from crewai_tools import BaseTool
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import OnlinePDFLoader, WebBaseLoader, CSVLoader, EverNoteLoader, PyMuPDFLoader, TextLoader, UnstructuredEmailLoader, UnstructuredEPubLoader, UnstructuredHTMLLoader,UnstructuredMarkdownLoader,UnstructuredODTLoader,UnstructuredPowerPointLoader,UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from pydantic import BaseModel, Field
from tqdm.autonotebook import tqdm, trange
from langchain.docstore.document import Document
from dotenv import load_dotenv

load_dotenv()

print("ENV", load_dotenv())

#Example queries, since the crewai only accepts 1 input, we then embed other variables on the input parameter
#E.g. on the below query, sdg and not_sdg are just placeholders for wether the query is about sdgs or not
"""
sdg,
NDC-SDG Connections for climate action,
Climate change actitities for awareness,
National determined contributions for climate change,
climate change mitigation activities,
LINKING NATIONALLY DETERMINED CONTRIBUTIONS (NDCS) AND SUSTAINABLE DEVELOPMENT GOALS (SDGS)
"""
class FixedGreyLitQueryToolSchema(BaseModel):
    """Input for GreyLitTool."""
    search_query: str = Field(..., description="Mandatory query you like to search from internet")
    
class GreyLitToolSchema(FixedGreyLitQueryToolSchema):
    """Input for GreyLitTool."""
    query_name: str = Field(
        ..., description="Mandatory valid query name you would like to search"
    )
    

class GreyLiteratureTool(RagTool):

    name:str = "Grey Literature Search Tool"
    description: str = "This tool is searches the internet for grey literature articles"
    args_schema: Type[BaseModel] = GreyLitToolSchema
    
    def __init__(self, query_name: Optional[str] = None, **kwargs):
        query_results = []
        json_data = []
        SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
        query_results = []
        n_path = 'C:\\Users\\MandlenkosiNgwenya\\Documents\\experiments\\output_main\\'
        json_data = []
        
        print(f"INITIALIZED >>> Curent Path: {n_path}")
        super().__init__(**kwargs)
                    
    # name: str = "JsonBuilderTool"
    @staticmethod
    @tool("Data Collector Tool")
    def _run(queries: str):
        """Use the given queries to search the internet and retrieve results"""
        try:
            
            query_list = queries.split(",")
            #Take only the first item on the list
            selector = re.sub('\n', '', query_list[0])
            
            if selector == 'not_sdg':
                filename = f"data_output"
            else:
                filename = f"{selector}_data_output"
                
            start_time = time()
            
            # only query from query 1 instead of index 0 which is just flag or placeholder
            for query in query_list[:1]:
                query = re.sub('\n', '', query)
                try:
                    print(query)
                    print("*******************************************************************************************************************************************************************")
                    GreyLiteratureTool.retrieve_ds(query)
                    # self.save_json(filename)
                except Exception as ex:
                    print(f'Error in @ query: {query}. {ex}')
                    continue
                
            end_time = time()
            time_diff = end_time - start_time
            print(f'PROGRAM TOOK {time_diff} SECONDS')
            GreyLiteratureTool.create_json(filename) 
            GreyLiteratureTool.create_csv(filename)
            return GreyLiteratureTool.json_data
        except Exception as ex:
            end_time = time()
            time_diff = end_time - start_time
            print(f"Error happened for query=> {ex}")
            print(f"Time taken to execute {time_diff}")
    
    @staticmethod
    def parse_to_sortedjson(item):
        try:
            jdata = json.loads(item)
            sorted_data = sorted(jdata['organic'], key=lambda x:x['title'])
            for i in range(len(sorted_data)):
                sorted_data[i]['query'] = jdata['searchParameters']['q']
                sorted_data[i]['title'] = jdata['organic'][i]['title']
                sorted_data[i]['link'] = jdata['organic'][i]['link']
                sorted_data[i]['position'] = jdata['organic'][i]['position']
                sorted_data[i]['snippet'] = jdata['organic'][i]['snippet']
                
                GreyLiteratureTool.json_data.append(sorted_data[i])
        # output_csv(sorted_data, filename)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
        
    @staticmethod
    def retrieve_ds(query):
        try:
            conn = client.HTTPSConnection("google.serper.dev")
            payload = json.dumps({
            "q": query,
            "gl": "za",
            "num": 101
            })
            headers = {
            'X-API-KEY': GreyLiteratureTool.SERPER_API_KEY,
            'Content-Type': 'application/json'
            }
            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()
            data = res.read()
            item = str(data.decode("utf-8"))
            # self.query_results.append(data)
            GreyLiteratureTool.parse_to_sortedjson(item)
            # print(data.decode("utf-8"))
        except Exception as ex:
            print(f'ERROR OCCURED WHILE SEARCHING WEB::: {ex}')

    @staticmethod
    def create_json(filename):
        with open(GreyLiteratureTool.n_path + filename +'.json', 'w') as f:
            json.dump(GreyLiteratureTool.json_data, f)
    
    @staticmethod
    def create_csv(filename):
        try:
            csv_file = GreyLiteratureTool.n_path + filename + '.csv'
            fieldnames = ['title', 'link', 'snippet', 'position', 'query']
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='|')
                writer.writeheader()
                for dataItem in GreyLiteratureTool.json_data:
                    try:
                        if isinstance(dataItem, dict):
                            filtered_data = {key: dataItem[key] for key in fieldnames if key in dataItem}
                            writer.writerow(filtered_data)
                        else:
                            dataDict = { 'title': dataItem['title'] , 'link': dataItem['link'], 'snippet': dataItem['snippet'], 'position': dataItem['position'], 'query': dataItem['query'] }
                            writer.writerows(dataDict)
                    except Exception as ex:
                        print(f"ERROR OCCURED WHILE ON ITEM::: {dataItem}. Original Exception: {ex}")
                        continue
        
            print(f"CSV file '{csv_file}' has been created successfully.")
            return True
        except Exception as ex:
            print(f"ERROR WHILE CREATING CSV FILE <<{csv_file}>>. ERROR:{ex}")
            return False