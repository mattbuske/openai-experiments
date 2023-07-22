from openai.embeddings_utils import get_embedding
from tenacity import retry, wait_random_exponential, stop_after_attempt

import openai
import os
import shutil
import PyPDF2
import tiktoken
import ingest.emails as emails
import ingest.chunk as chunk
import utils.file as utils

# Initial Context
context = "You are a helpful assistant that ingests files and stores it locally. "
fact_summary_context = "Can you list every single fact in the following text retaining every fact in a lossless list along with it's source along with as much contextual nuance as possible: "

#word_count = 1500
# gpt-3.5-turbo-16k - Possible
general_model = "gpt-3.5-turbo"
embedding_model = "text-embedding-ada-002"

# account for the initial prompt tokens and a buffer TODO: Research best practices on chunking
token_buffer=200
hard_context_limit=4097
context_token_limit = (hard_context_limit-token_buffer)

# get the number of tokens from a string using tiktoken
def num_tokens_from_string(string: str) -> int:
    encoding_name = tiktoken.encoding_for_model(general_model)
    encoding = tiktoken.get_encoding(encoding_name.name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Get the File Type 'Header'
def get_file_header(file_type):
    return f"File Type: {file_type} "

# Splits the extracted text from a pdf into chunks
def chunk_pdf(file_path):

    with open(file_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        # Initialize an empty string to store the extracted text
        extracted_text = get_file_header('PDF')
        num_pages = len(pdf_reader.pages)
        page_num = 1
        # Iterate through each page of the PDF
        for page_num in range(num_pages):
            # Extract the text from the current page
            page = pdf_reader.pages[page_num]
            text = page.extract_text()  # Read the file and replace newlines with spaces
            page_num+=1
            # Append the extracted text to the result
            extracted_text += text

    return chunk.get_chunks(extracted_text)

# Splits the extracted text from a text file into chunks
def chunk_text_file(file_path):
    
    extracted_text = get_file_header('Text')

    with open(file_path, 'r') as file:
        extracted_text = file.read()  # Read the file and replace newlines with spaces

    return chunk.get_chunks(extracted_text)

def get_user_message(text):
    return {"role": "user", "content": text}

# Get Vector Objects
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_vector_objects(chunk, meta):

    # Get Full Embeddings for the Text Chunk
    embeds = get_embedding(chunk,engine=embedding_model)
    # Return the vector object to store in memory
    return {
        'uid': str(id(embeds)),
        'embeddings': embeds,
        'meta': meta
    }

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def get_chat_completion(
    messages,
    model=general_model
):
    response = {}
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )
    choices = response["choices"]  # type: ignore
    completion = choices[0].message.content.strip()
    return completion

# Add a vector information to the 'Brain'
def add_to_vector_memory(text,source):
    # Get the basic text vector
    obj = get_vector_objects(text, {
        'title': source,
        'file': source,
        'text': text 
    })
    utils.append_to_json_file('knowledge.json',obj)

# Processes documents in a folder and it's subfolder
def process_documents(source_folder, destination_folder):

    # Loop through the files in the source folder
    for filename in os.listdir(source_folder):

        # Ignore readme files as those are for the developers
        if "Readme.md" != filename:

            # Get the full path of the current file
            source_file = os.path.join(source_folder, filename)

            # Check if the current item is a directory
            if os.path.isdir(source_file):
                # make sure the corresponding directory exists in the 'Learnt' folder
                utils.create_directory(os.path.join(destination_folder,filename))
                # process the directories
                process_documents(os.path.join(source_folder,filename),os.path.join(destination_folder,filename))

            # Check if the current item is a file
            if os.path.isfile(source_file):
                file_extension = utils.get_file_extension(source_file)
                if file_extension == '.pdf':
                    chunks = chunk_pdf(source_file)
                elif file_extension == '.eml':
                    chunks = emails.chunk_email(source_file)
                else:
                    chunks = chunk_text_file(source_file)
            
                for i in chunks:
                    # We should add it to memory here
                    add_to_vector_memory(i,filename)
                    
                # Move the processed document to the destination folder
                destination_file = os.path.join(destination_folder, filename)
                shutil.move(source_file, destination_file)
                print(f"Moved {filename} to {destination_folder}")