from openai.embeddings_utils import get_embedding
from tenacity import retry, wait_random_exponential, stop_after_attempt

import openai
import json
import os
import shutil
import PyPDF2
import tiktoken

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

# Constants
CHUNK_SIZE = 200  # The target size of each text chunk in tokens
MIN_CHUNK_SIZE_CHARS = 350  # The minimum size of each text chunk in characters
MIN_CHUNK_LENGTH_TO_EMBED = 5  # Discard chunks shorter than this
EMBEDDINGS_BATCH_SIZE = int(os.environ.get("OPENAI_EMBEDDING_BATCH_SIZE", 128))  # The number of embeddings to request at a time
MAX_NUM_CHUNKS = 10000  # The maximum number of chunks to generate from a text


# get the number of tokens from a string using tiktoken
def num_tokens_from_string(string: str) -> int:
    encoding_name = tiktoken.encoding_for_model(general_model)
    encoding = tiktoken.get_encoding(encoding_name.name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Create a directory only if it doesn't already exist
def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

# Append the information to the JSON Object
def append_to_json_file(file_path, new_objects):
    # Read the existing data from the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Append new objects to the existing data
    data.append(new_objects)
    
    # Write the updated data back to the JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Split the text into chunks
def combine_chunks(chunks, extracted_text):

    encoding_name = tiktoken.encoding_for_model(general_model)
    tokenizer = tiktoken.get_encoding(encoding_name.name)

    # Tokenize the text
    tokens = tokenizer.encode(extracted_text, disallowed_special=())
    num_chunks = 0

    # Loop until all tokens are consumed
    while tokens and num_chunks < MAX_NUM_CHUNKS:
        
        # Take the first chunk_size tokens as a chunk
        chunk = tokens[:CHUNK_SIZE]

        # Decode the chunk into text
        chunk_text = tokenizer.decode(chunk)

        # Skip the chunk if it is empty or whitespace
        if not chunk_text or chunk_text.isspace():
            # Remove the tokens corresponding to the chunk text from the remaining tokens
            tokens = tokens[len(chunk) :]
            # Continue to the next iteration of the loop
            continue

        # Find the last period or punctuation mark in the chunk
        last_punctuation = max(
            chunk_text.rfind("."),
            chunk_text.rfind("?"),
            chunk_text.rfind("!"),
            chunk_text.rfind("\n"),
        )

        # If there is a punctuation mark, and the last punctuation index is before MIN_CHUNK_SIZE_CHARS
        if last_punctuation != -1 and last_punctuation > MIN_CHUNK_SIZE_CHARS:
            # Truncate the chunk text at the punctuation mark
            chunk_text = chunk_text[: last_punctuation + 1]

        # Remove any newline characters and strip any leading or trailing whitespace
        chunk_text_to_append = chunk_text.replace("\n", " ").strip()

        if len(chunk_text_to_append) > MIN_CHUNK_LENGTH_TO_EMBED:
            # Append the chunk text to the list of chunks
            chunks.append(chunk_text_to_append)

        # Remove the tokens corresponding to the chunk text from the remaining tokens
        tokens = tokens[len(tokenizer.encode(chunk_text, disallowed_special=())) :]

        # Increment the number of chunks
        num_chunks += 1

    # Handle the remaining tokens
    if tokens:
        remaining_text = tokenizer.decode(tokens).replace("\n", " ").strip()
        if len(remaining_text) > MIN_CHUNK_LENGTH_TO_EMBED:
            chunks.append(remaining_text)

    return chunks

# Splits the extracted text from a pdf into chunks
def chunk_pdf(file_path):

    with open(file_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        # Initialize an empty string to store the extracted text
        extracted_text = ''
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

    # Return an empty list if the text is empty or whitespace
    if not extracted_text or extracted_text.isspace():
        return []

    chunks = combine_chunks([], extracted_text)

    return chunks

# Splits the extracted text from a text file into chunks
def chunk_text_file(file_path):
    
    with open(file_path, 'r') as file:
        extracted_text = file.read()  # Read the file and replace newlines with spaces

    # Return an empty list if the text is empty or whitespace
    if not extracted_text or extracted_text.isspace():
        return []

    chunks = combine_chunks([], extracted_text)
    
    return chunks

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
    append_to_json_file('knowledge.json',obj)

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
                create_directory(os.path.join(destination_folder,filename))
                # process the directories
                process_documents(os.path.join(source_folder,filename),os.path.join(destination_folder,filename))

            # Check if the current item is a file
            if os.path.isfile(source_file):
                file_extension = os.path.splitext(source_file)[1].lower()
                if file_extension == '.pdf':
                    chunks = chunk_pdf(source_file)
                else:
                    chunks = chunk_text_file(source_file)
            
                for i in chunks:
                    
                    # Add full text to vector memory
                    add_to_vector_memory(i,filename)
                    # Get Just the Facts from the text
                    #fact_summary = get_chat_completion([get_user_message(fact_summary_context+i)])
                    # Add just the facts to the vector memory
                    #add_to_vector_memory(fact_summary,filename)
                    
                # Move the processed document to the destination folder
                destination_file = os.path.join(destination_folder, filename)
                shutil.move(source_file, destination_file)
                print(f"Moved {filename} to {destination_folder}")

# Process the Documents
process_documents('Learnin', 'Learnt')




