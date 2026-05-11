import time

from huggingface_hub import cached_download
# from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from langchain.chains import RetrievalQA

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.llms import ollama
from langchain_groq import ChatGroq

# import chromadb
import os
import argparse
import textwrap
from dotenv import load_dotenv

# from evaluation_sdg_crew.tools.grey_lit_tool import GreyLiteratureTool
load_dotenv(dotenv_path="D:/Dev/BusinessApps/citizensci_experim/src/evaluation_sdg_crew/.env")

# model = os.environ.get("MODEL", "mistral")
# model_name = os.environ.get("MODEL", "llama3-8b-8192")
model_name = os.environ.get("MODEL", "llama3.1")
# groq_api_key = os.environ.get("GROQ_API_KEY")

# For embeddings model, the example uses a sentence-transformers model
# https://www.sbert.net/docs/pretrained_models.html 
# "The all-mpnet-base-v2 model provides the best quality, while all-MiniLM-L6-v2 is 5 times faster and still offers good quality."
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',16))
model_name = os.environ.get("MODEL", "llama3-8b-8192")
model_name_best = os.environ.get("MODEL", "llama-3.1-70b-versatile") 
# source_docs = "C:\\Users\\MandlenkosiNgwenya\\chroma_db"
# source_docs = "C:\\Users\\MandlenkosiNgwenya\\policy_db"
# source_docs = "C:\\Users\\MandlenkosiNgwenya\\policy_db_sdg_lit"
# source_docs = "C:\\Users\\MandlenkosiNgwenya\\researcher_db3"
source_docs = "C:\\Users\\MandlenkosiNgwenya\\researcher_db_abstracts_1.1"
TOKENIZERS_PARALLELISM="false"

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=source_docs, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # llm = ollama.Ollama(model=model_name, callbacks=callbacks)
    llm = ChatGroq(
            model_name=model_name_best)
    # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not args.hide_source, verbose=True, chain_type_kwargs={"verbose": True})

    # Load embeddings outside the loop if they don't change
    # Process queries in batches for better performance
    # PROMPT "From the list of documents available. What top 5 documents closely relate with this document? "Mapping Climate Actions and the Sustainable Development Goals: A Review of Current Approaches"
    # rag_template = """Use only the documents given to you and do not use external documents or sources. Keep your answers short, do not give a summary of the articles, just mention the title and the article's accurate metadata which is provided with the documents. If you don't know the answer simply say *As a <<<Research Reviewer>>> I don't have information on that*.
    
    # rag_template = """Use only the documents given to you and do not use external documents or sources. Your task is to extract climate mitigation projects from the text and location of those projects.
    #                   Use the example given to learn what to extract from the text.
    #                   Keep your answers short, do not give a summary of the articles. 
    #                   Show a valid document source as a valid link which is provided on the document metadata.
    #                   Do not duplicate the sources.
    #                   If you don't know the answer simply say *As a Researcher Reviewer I don't have information on that. 
    
    # rag_template = """Answer the questions based on the given documents, also refer to the local source where you found you're answers. Your task is to extract climate projects from the text and location of those projects.
    #                   Keep your answers short, do not give a summary of the articles.
    #                   If you don't know the answer simply say *As a Researcher Reviewer I don't have information on that. 
    # rag_template = """Answer the questions based on the given documents. Keep your answers short, do not give a summary of the articles. 
    #                     If you don't know the answer simply say *As a Researcher Reviewer I don't have information on that.
    #                     Only give results from the South African context, with results provide a valid URL from the given documents.
    #                     When you are asked to list, at least provide a list of 15 relevant documents with their URL.
    # rag_template_outcomes = """Analyze the following text and classify it into one or more of the following categories: 
    #                 "Improved Education," "Improved Awareness-raising," "Improved Human Capacity," or "Improved Institutional Capacity." 
    #                  After classification, provide a brief explanation for your choice.
    # rag_template = """Analyze the following text and classify it into one or more of the following categories: 
    #                 "Improved Education," "Improved Awareness-raising," "Improved Human Capacity," or "Improved Institutional Capacity." 
    #                  After classification, provide a brief explanation for your choice.
    rag_template = """Analyze the following text and classify it into one or more of the following intervention categories: 
                        Intervention:"Tree planting," "Coastal conservation," or "Wetland restoration"\n.
                        Classify to the following applicable, Outcome: "Improved Education," "Improved Awareness-raising," "Improved Human Capacity," or "Improved Institutional Capacity." \n.
                        Extract project/initiative or programme specified\n.
                        Extract organizations mentioned\n.
                        Extract People Type mentioned, e.g. learners or scholars or educators\n.
                        Extract Places mentioned, e.g. South Africa or Gauteng\n.
                        Extract Impact made by the project/initiative, Get the impact from the study findings/output/discussions/abstract.Example of impact: <<<100 trees were planted at these schools which helped with climate change mitigation...>>>\n. 
                        After classification, provide a brief explanation for your choice\n.
                        When the information is missing, do say so.
                        
    {context}
    Question: {question}
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_template)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    # activate/deactivate the streaming StdOut callback for LLMs
    # callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    # llm = Ollama(model=model, callbacks=callbacks)

    # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source, verbose=True,
    # chain_type_kwargs={
    #     "verbose": True,
    # })
    # Interactive questions and answers
    while True:
        query = input("\n\nAsk Reviewer Assistant: ")
        x_query = f"{query}. Show a valid documents' unique URLs which was provided for in the documents. Use only the given documents. URLs must be unique and should not repeat."   
        example = f"""
            Example: <<<
                    In India, the Slum Networking Project has provided basic services such as water supply, sanitation, drainage, solid waste management, street lighting, and paved roads to over 100,000 slum dwellers, while also promoting low-carbon solutions such as biogas plants, solar panels, and rooftop gardens.
                    Project: Slum Networking Project
                    Country/Location: India
                    Impact: provided basic services such as water supply sanitation, drainage, solid waste management, street lighting, and paved roads to over 100,000 slum dwellers; 
                            promoting low-carbon solutions such as biogas plants, solar panels, and rooftop gardens.
                    URL: https://www.mdpi.com/1996-1073/12/14/2798
                    >>>
                    Use the example given to learn what to extract from the text.
                    
                """
        # query = str(x_query + "\n\n" + example)
        # query = str(x_query)
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        print("Started", start)
        print("\n\n> Question:")
        response = rag_chain.invoke(query)
        print(textwrap.fill(response, width=80))
        end = time.time()
        print("Completed after", (end - start)/60*60, " seconds")
        # answer, docs = res['result'], [] if args.hide_source else res['source_documents']


    # while True:
    #     query = input("\nEnter a query: ")
    #     if query == "exit":
    #         break
    #     if query.strip() == "":
    #         continue

    #     # Get the answer from the chain
    #     start = time.time()
    #     res = qa(query)
    #     end = time.time()
    #     print("Completed after", (end - start)/60*60, " seconds")
    #     print("RESOURCE", res)
    #     answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        
    #     # Print the result
    #     print("\n\n> Question:")
    #     print(query)
    #     print(answer)