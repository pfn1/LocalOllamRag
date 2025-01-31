
# NOTE: For all imports double check the updated import statements from documentation
# Multiple cases where import statements were old and does not work

# useful for command line arguments
import argparse


# import Ollama LLM
from langchain_ollama.llms import OllamaLLM

# import Chroma
from langchain_chroma import Chroma

# import chat template
from langchain.prompts import ChatPromptTemplate  #might be old version
from langchain_core.prompts import ChatPromptTemplate

# import the embedding function from the embedding_functions.py file
from embedding_functions import get_embedding_function



# db path
CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# main function for querying questions through the command line
def main():
   parser = argparse.ArgumentParser()
   parser.add_argument(
      "query_text",
      type=str,
      help="The question to ask the model.",
   )
   args = parser.parse_args()
   query_text = args.query_text
   query_rag(query_text) # call the query_rag function with the query_text as the argument

# Function to query the RAG database
def query_rag(query_text: str):
   # call the embedding function to get the embeddings
   embedding_function = get_embedding_function()
   # load the db
   db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

   # search the db with the query text and get the results
   # the results are chunks of text that was stored in the db before
   # k=5 means we want the top 5 most similar chunks based on similarity_search_with_score function
   # there are other search functions like similarity_search that can be used
   results = db.similarity_search_with_score(query_text, k=5)

   # format the chunks from results
   context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
   prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
   prompt = prompt_template.format(context=context_text, question=query_text)


   # call the Ollama LLM model to get the response
   # try different models like "mistral" or "llama3.1"
   model = OllamaLLM(model="mistral")
   # model = Ollama(model="llama3.1")
   # execute the prompt and get the response
   response_text = model.invoke(prompt)

   # format the response and output to console
   sources = [doc.metadata.get("id", None) for doc, _score in results]
   formatted_response = f"Response: {response_text}\nSources: {sources}"
   print(formatted_response)
   return response_text


if __name__ == "__main__":
   main()