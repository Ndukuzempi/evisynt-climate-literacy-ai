import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path="D:/Dev/BusinessApps/citizensci_experim/src/evaluation_sdg_crew/.env")

print("ENV", load_dotenv(dotenv_path="D:/Dev/BusinessApps/citizensci_experim/src/evaluation_sdg_crew/.env"))
# os.environ["SERPER_API_KEY"] = "cd28110cb9c7d3a4269ff10751b23a040a8413d8"

from evaluation_sdg_crew.crew import SDGTargetEvaluationAgents

country= "South Africa"
all_queries = """
    Climate change education {country}.
    Engagement on global warming {country}.
    Citizen activism on climate change impact {country}.
    {country} climate change awareness campaigns.
    Impact of climate change education {country}.
    Community-based climate change projects {country}.
    Citizen engagement climate change {country}.
    Climate change advocacy {country}.
    Impact of climate change education {country}.
    {country}n civil society climate change engagement.
"""
queries = """
    Climate change education {country}.
"""
# mapping_structure = 'config/task_train_case1.yaml'


def run():
    inputs ={
        'event_topic': "citizen engagements and events on climate change",
        'event_description': "citizen engagements and education on global warming or climate change in South Africa",
        'country': "South Africa",
        'period': 'past 3 years',
        'queries': queries,
        'sdg_target': 'SDG target 13.3'
        # 'path_to_example_data': './data/sdg_map.csv'
    }
    
    SDGTargetEvaluationAgents().crew().kickoff(inputs=inputs)

def train():

    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'mapping_structure': mapping_structure
    }
    
    try:
        SDGTargetEvaluationAgents().crew().train(n_iterations=int(sys.argv[1]), inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

if __name__ == '__main__':
    run()
