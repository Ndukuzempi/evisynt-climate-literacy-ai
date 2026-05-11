import os
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, crew, task
from langchain_groq import ChatGroq
from crewai_tools import ScrapeWebsiteTool, SerperDevTool, FileReadTool
from pydantic import BaseModel, Field
from typing import List
from langchain_community.llms import ollama
# from langchain.agents import load_tools
from crewai_tools import tool


from dotenv import load_dotenv

# from evaluation_sdg_crew.tools.grey_lit_tool import GreyLiteratureTool
load_dotenv(dotenv_path="D:/Dev/BusinessApps/citizensci_experim/src/evaluation_sdg_crew/.env")

# from evaluation_sdg_crew.tools.data_collection_json_builder import GreyLitCollector

queries = """not_sdg,
climate change education South Africa,
engagement on global warming South Africa,
citizen activism on climate change impact South Africa,
South Africa climate change awareness campaigns,
impact of climate change education South Africa,
community-based climate change projects South Africa,
Citizen engagement climate change South Africa,
climate change advocacy South Africa,
impact of climate change education South Africa,
South African civil society climate change engagement"""

search_tool = SerperDevTool(n_results=50, country="za", tbs='qdr:y3' )
scrape_tool = ScrapeWebsiteTool()
# grey_lit_tool = GreyLiteratureTool()
# grey_lit_tool = grey_lit_request.get_data(queries)
# file_read_tool = FileReadTool(
# 	file_path='./data/mapping_examples.md',
# 	description='A tool to read examples of mapping to SDG'
# )

class QueryItem(BaseModel):
    title: str = Field(..., description="Title of the searched article")
    url: str = Field(..., description="Valid URL of the article")

class QueryResults(BaseModel):
    query: str = Field(..., description="Search query")
    results : List[QueryItem] = Field(..., description="List of search results by title and url")

file_read_tool = FileReadTool(file_path='./data/mapping_examples.md')
# read_write_md_files_tool = MDXSearchTool(mdx='./data/mapping_examples.md')
model_name = os.environ.get("MODEL", "llama3-8b-8192")
model_name_best = os.environ.get("MODEL", "llama-3.1-70b-versatile") 
# model_name = os.environ.get("MODEL", "llama3-70b-8192")

@CrewBase
class SDGTargetEvaluationAgents():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'
    
    @tool("Ask Human follow up questions")
    def ask_human(question: str) -> str:
        """Ask human to enter a query"""
        print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
        contents = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == "q":
                break
            contents.append(line)
        return "\n".join(contents)
        
    def get_input() -> str:
        print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
        contents = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == "q":
                break
            contents.append(line)
        return "\n".join(contents)
    
    def __init__(self) -> None:
        self.groq_llm = ChatGroq(temperature=0, model_name=model_name)
        self.best_model = ChatGroq(temperature=0, model_name=model_name_best)
        self.locallm = ollama.Ollama(model="llama3.1", base_url="http://localhost:11434")
        # self.human_tools = load_tools(["human", "ddg-search"], llm=self.locallm, input_func=self.get_input)

        
    @agent
    def assistant_researcher(self):
        return Agent(
            config=self.agents_config['researcher'],
            llm=self.best_model,
        )

    # @agent
    # def reviewer(self):
    #     return Agent(
    #         config=self.agents_config['reviewer'],
    #         llm=self.best_model
    #     )

    # @agent
    # def sdg_specialist(self):
    #     return Agent(
    #         config=self.agents_config['sdg_specialist'],
    #         llm=self.locallm,
    #         # tools=self.human_tools
    #     )

    # @agent
    # def sector_specialist(self):
    #     return Agent(
    #         config=self.agents_config['sector_specialist'],
    #         llm=self.best_model,
    #         # llm=self.locallm
            
    #     )
        
    @task
    def reseach_assistant_task(self) -> Task:
        return Task(
            config=self.tasks_config['reseach_assistant_task'],
            agent=self.assistant_researcher(),
            # tools=[search_tool, scrape_tool]
            # human_input=True,
            tools=[search_tool],
            # output_json=QueryResults,
            output_file="./data/researched_data.md",
            # output_file="./data/researched_data.json",
        
            )
        
        
    # @task
    # def reviewer_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['reviewer_task'],
    #         agent=self.reviewer(),
    #         # output_file="./data/screend_data.md",
    #         output_json=QueryResults,
    #         output_file="./data/screend_data.json",

    #     )
        
    # @task
    # def sdg_specialist_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['sdg_specialist_task'],
    #         agent=self.sdg_specialist(),
    #         tools=[search_tool],
    #         output_file="./data/sdg_countries_data.md",

    #     )
        
    # @task
    # def sector_specialist_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['sector_specialist_task'],
    #         agent=self.sector_specialist(),
    #         context=[self.tasks_config['reviewer_task'], self.tasks_config['reseach_assistant_task']],
    #         tools=[search_tool, scrape_tool],
    #         output_file="./data/sdg_mapped_data.md",
    #     )
        
   

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents = self.agents,
            tasks = self.tasks,
            process = Process.sequential,
            verbose = 2,
            output_log_file=True
        )