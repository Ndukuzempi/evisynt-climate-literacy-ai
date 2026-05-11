# AI Crew for M&E For SDG

**DISCALIMER** This is a research project done for the possible use case of LLM agents in the M&E space.

## Introduction
This project demonstrates the use of the CrewAI framework to automate the M&E evidence synthesis process. CrewAI orchestrates autonomous AI agents, enabling them to collaborate and execute complex tasks efficiently.

By [@ndukuzempi](https://x.com/ndukuzempi)

- [CrewAI Framework](#crewai-framework)
- [Running the script](#running-the-script)
- [Details & Explanation](#details--explanation)
- [Contributing](#contributing)
- [Support and Contact](#support-and-contact)
- [License](#license)

## CrewAI Framework
CrewAI is designed to facilitate the collaboration of role-playing AI agents. In this example, these agents work together to streamline the M&E and SDG Mapping to target process

## Running the Script

***DISCALIMER:** This project uses Selenium to etract text from websites, and it's meant only as an research, using this for real-world applications may violate websites's terms of service.*

- **Configure Environment**: Copy `.env` and set up the environment variables for [Groq](https://console.groq.com/keys) and other tools as needed. Key will be revoked after once the project is shared on github
- **Install Dependencies**: Run `poetry lock && poetry install`.
- **Customize**: Modify `src/evaluation_sdg_crew/main.py` to add custom inputs for your agents and tasks.
- **Customize Further**: Check `src/evaluation_sdg_crew/config/agents.yaml` to update your agents and `src/evaluation_sdg_crew/config/tasks.yaml` to update your tasks.
- **Custom Tools**: You can find custom tools at `evaluation_sdg_crew/src/tools/`.
- **Execute the Script**: Run `poetry run evaluation_sdg_crew` and input your project details.

## Details & Explanation
- **Running the Script**: Execute `poetry run evaluation_sdg_crew`. The script will leverage the CrewAI framework to automate M&E tasks and generate a detailed report.
- **Key Components**:
  - `src/evaluation_sdg_crew/main.py`: Main script file.
  - `src/evaluation_sdg_crew/crew.py`: Main crew file where agents and tasks come together, and the main logic is executed.
  - `src/evaluation_sdg_crew/config/agents.yaml`: Configuration file for defining agents.
  - `src/evaluation_sdg_crew/config/tasks.yaml`: Configuration file for defining tasks.
  - `src/evaluation_sdg_crew/tools`: Contains tool classes used by the agents.

## License
This project is released under the MIT License.