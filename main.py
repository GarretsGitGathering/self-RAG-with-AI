import os  # Library for interacting with the operating system
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Module for splitting text into chunks
from langchain_community.document_loaders import WebBaseLoader  # Module for loading documents from the web
from langchain_community.vectorstores import Chroma  # Module for storing and querying vectors
from langchain_openai import OpenAIEmbeddings  # Module for working with OpenAI embeddings
#from GraphState import *

import pprint

from langgraph.graph import END, StateGraph
from function import question_relevence_score, generate, grade_documents, transform_query, decide_to_generate, prepare_for_final_grade, grade_generation_v_documents, grade_generation_v_question
from typing import Dict, TypedDict
from lookup import initial_lookup

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "*** YOUR API KEY ***"    

# define the question
question = "what are the minions from despicable me?"

# lookup a list of URLs to load documents from
urls = initial_lookup(question, 5)
print(urls)

# Load documents from the web
docs = [WebBaseLoader(url).load() for url in urls]

# Flatten the list of lists into a single list
docs_list = [item for sublist in docs for item in sublist]

# Initialize an empty list to store split documents
result = []

# Loop through each document and split it into chunks
for sublist in docs:
    for item in sublist:
        result.append(item)

# Initialize a text splitter with specified parameters
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

# Split the documents into smaller chunks
doc_splits = text_splitter.split_documents(docs_list)

# Create a vector store from the split documents
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
)

# Convert the vector store into a retriever
retriever = vectorstore.as_retriever()      # https://python.langchain.com/docs/get_started/quickstart#retrieval-chain


def retrieve(state): 
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVING SOURCES---")

    state_dict = state["keys"]
    question = state_dict["question"]
    documents = retriever.get_relevant_documents(question)          #should probably check relevant documents individually
    return {"keys": {"documents": documents, "question": question}}

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]

def create_workflow():
    # Initialize the workflow graph with the initial state
    workflow = StateGraph(GraphState)

    # Define the nodes in the workflow graph
    workflow.add_node("retrieve", retrieve)  # Node to retrieve documents
    workflow.add_node("grade_documents", grade_documents)  # Node to grade documents
    workflow.add_node("generate", generate)  # Node to generate an answer
    workflow.add_node("transform_query", transform_query)  # Node to transform the query
    workflow.add_node("prepare_for_final_grade", prepare_for_final_grade)  # Node for preparing the final grade

    # Build the graph by connecting the nodes
    workflow.set_conditional_entry_point(   # Set the starting point of the graph at the question_relevence_score node
        question_relevence_score,
        {
            "yes": "retrieve",
            "no": END
        }
    )
    workflow.add_edge("retrieve", "grade_documents")  # Connect retrieve node to grade_documents node
    workflow.add_conditional_edges(  # Add conditional edges based on the output of grade_documents
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",  # If decide_to_generate returns "transform_query", go to transform_query node
            "generate": "generate",  # If decide_to_generate returns "generate", go to generate node
        },
    )
    workflow.add_edge("transform_query", "retrieve")  # Loop back to retrieve node from transform_query node
    workflow.add_conditional_edges(  # Add conditional edges based on the output of generate
        "generate",
        grade_generation_v_documents,
        {
            "supported": "prepare_for_final_grade",  # If generate is supported, go to prepare_for_final_grade node
            "not supported": "generate",  # If generate is not supported, loop back to generate node
        },
    )
    workflow.add_conditional_edges(  # Add conditional edges based on the output of prepare_for_final_grade
        "prepare_for_final_grade",
        grade_generation_v_question,
        {
            "useful": END,  # If prepare_for_final_grade is useful, end the workflow
            "not useful": "transform_query",  # If prepare_for_final_grade is not useful, go back to transform_query node
        },
    )

    # Compile the workflow into an executable application
    app = workflow.compile()
    return app


def main():
    inputs = {"keys": {"question": question, "source_count": 4, "generation":""}}

    app = create_workflow()

    config = {"recursion_limit": 25}
    for output in app.stream(inputs, config=config):
        for key, value in output.items():
            # Node
            pprint.pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint.pprint("---------------------------------------------")

    # Final generation
    if(value["keys"]["generation"]!=""):
        pprint.pprint(value["keys"]["generation"])
    else:
        pprint.pprint("\nplease only ask questions\n")


if __name__ == "__main__":
    main()
