#This file contains the embedding function that will be used to embed the text chunks
#Alternative embeddings are also in this file

# NOTE: For all imports double check the updated import statements from documentation
# Multiple cases where import statements were old and does not work

from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings


def get_embedding_function():
   """ embeddings = BedrockEmbeddings(
      credentials_profile_name="default", region_name="us-east-1"
   ) """
   # can use more embeddings like below
   embeddings = OllamaEmbeddings(model="nomic-embed-text")

   return embeddings

