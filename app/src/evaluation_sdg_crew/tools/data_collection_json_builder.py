import re, os, json, csv
import stat
from time import time
from http import client
from datetime import datetime
from crewai_tools import SerperDevTool
from crewai_tools import BaseTool
from langchain_core.retrievers import BaseRetriever
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


class GreyLitCollector:
    query_results = []
    json_data = []
    SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
    name = "This tool is used to build json from internet search results, this tool outputs json file as output"
    query_results = []
    n_path = 'C:\\Users\\MandlenkosiNgwenya\\Documents\\experiments\\output_main\\'
    json_data = []
        
    print(f"INITIALIZED >>> Curent Path: {n_path}")
                    
    # name: str = "JsonBuilderTool"
    # description: str = "This tool is used to build json from internet search results, this tool outputs json file as output"
    @staticmethod
    @tool("Data Collector Tool")
    def get_data(queries: str):
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
                    GreyLitCollector.retrieve_ds(query)
                    # self.save_json(filename)
                except Exception as ex:
                    print(f'Error in @ query: {query}. {ex}')
                    continue
                
            end_time = time()
            time_diff = end_time - start_time
            print(f'PROGRAM TOOK {time_diff} SECONDS')
            GreyLitCollector.create_json(filename) 
            GreyLitCollector.create_csv(filename)
            return GreyLitCollector.json_data
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
                
                GreyLitCollector.json_data.append(sorted_data[i])
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
            'X-API-KEY': GreyLitCollector.SERPER_API_KEY,
            'Content-Type': 'application/json'
            }
            conn.request("POST", "/search", payload, headers)
            res = conn.getresponse()
            data = res.read()
            item = str(data.decode("utf-8"))
            # self.query_results.append(data)
            GreyLitCollector.parse_to_sortedjson(item)
            # print(data.decode("utf-8"))
        except Exception as ex:
            print(f'ERROR OCCURED WHILE SEARCHING WEB::: {ex}')

    @staticmethod
    def create_json(filename):
        with open(GreyLitCollector.n_path + filename +'.json', 'w') as f:
            json.dump(GreyLitCollector.json_data, f)
    
    @staticmethod
    def create_csv(filename):
        try:
            csv_file = GreyLitCollector.n_path + filename + '.csv'
            fieldnames = ['title', 'link', 'snippet', 'position', 'query']
            
            with open(csv_file, 'w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='|')
                writer.writeheader()
                for dataItem in GreyLitCollector.json_data:
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