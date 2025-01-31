# This file is used load the contents of a pdf file 
# Then break the loaded data into manageable text chunks
# The text chunks will be embedded and then stored in a vector database
# The databsase will be used also during retrieval

# NOTE: For all imports double check the updated import statements from documentation
# Multiple cases where import statements were old and does not work

# import utility functions
import argparse # for parsing command line arguments
import os # used to work with files, directories, environment variables, and system commands.
import shutil # for file operations


# import for loading documents
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

# imports for spliting documents into text chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

# imports the embedding function from embedding_functions.py
# import for the Chroma db vector store
from embedding_functions import get_embedding_function
from langchain_chroma import Chroma

DATA_PATH = 'Data/'
CHROMA_PATH = "chroma"

# main function to run the script as a standalone script and not as an imported module
def main():
   #handles the command line arguments for the script
   parser = argparse.ArgumentParser()

   # adds a reset database option when running this script by itself and not as a module imported by another script
   parser.add_argument(
      "--reset",
      action="store_true",
      help="Reset the Chroma database by deleting the existing database.",
   )

   # Create (or update) the data store.
   documents = load_documents()
   chunks = split_documents_into_chunks(documents)
   add_to_chroma(chunks)  #load the chunks to chroma



# function to Load the documents from the Data directory
def load_documents():
   document_loader = PyPDFDirectoryLoader(DATA_PATH)
   return document_loader.load()

# Testing if data was loaded properly
""" 
documentTest = load_documents()
print('  (1)  Document Loaded Test !!!')
print(documentTest[0]) """


# function to break the loaded data into manageable text chunks
def split_documents_into_chunks(documents: list[Document]):
   text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=800,
      chunk_overlap=80,
      length_function=len,
      is_separator_regex=False,
   )
   return text_splitter.split_documents(documents)

# Testing if data was split into chunks
""" 
chunks = split_documents_into_chunks(documentTest)
print('  (2)  Document Split Test !!!')
print(chunks[0]) """

# function to store the embedded text chunks in a vector database
# call the get_embedding_function() function here to get the embedding function
def add_to_chroma(chunks: list[Document]):
   # Load the existing database.
   db = Chroma(
      persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
   )

   # Calculate Page IDs.
   chunks_with_ids = calculate_chunk_ids(chunks) # call the helper function

   # Add or Update the documents.
   existing_items = db.get(include=[])  # IDs are always included by default
   existing_ids = set(existing_items["ids"])
   print(f"Number of existing documents in DB: {len(existing_ids)}")

   # Only add documents that don't exist in the DB.
   new_chunks = []
   for chunk in chunks_with_ids:
      if chunk.metadata["id"] not in existing_ids:
         new_chunks.append(chunk)

   # Branch based on whether there are new documents to add by checking the length of new_chunks list
   if len(new_chunks):
      print(f"👉 Adding new documents: {len(new_chunks)}")
      new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
      db.add_documents(new_chunks, ids=new_chunk_ids)
   else:
      print("✅ No new documents to add")


# helper function to calculate the chunk ids
def calculate_chunk_ids(chunks):

   # This will create IDs like "data/monopoly.pdf:6:2"
   # Page Source : Page Number : Chunk Index

   last_page_id = None
   current_chunk_index = 0

   for chunk in chunks:
      source = chunk.metadata.get("source")
      page = chunk.metadata.get("page")
      current_page_id = f"{source}:{page}"

      # If the page ID is the same as the last one, increment the index.
      if current_page_id == last_page_id:
         current_chunk_index += 1
      else:
         current_chunk_index = 0

      # Calculate the chunk ID.
      chunk_id = f"{current_page_id}:{current_chunk_index}"
      last_page_id = current_page_id

      # Add it to the page meta-data. Adding a new column called "id" in metadata of the chunk with the chunk_id
      chunk.metadata["id"] = chunk_id

   return chunks

# helper function to clear the database if the --reset flag is passed in the command line 
# when running the script as a standalone script
def clear_database():
   if os.path.exists(CHROMA_PATH):
      shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
   main()