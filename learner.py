
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from openai.embeddings_utils import get_embedding

from langchain.schema import AIMessage,HumanMessage,SystemMessage

import json
import os
import shutil
import PyPDF2

#word_count = 1500
# gpt-3.5-turbo-16k - Possible
chat = ChatOpenAI(temperature=0,model='gpt-3.5-turbo')

messages = [
    SystemMessage(content="You are a helpful assistant that ingests files and stores it locally"),
]

# this is 40 tokens towards the limit
summary_prompt = "Please provide a concise yet comprehensive summary of the following text with the following constraints in order of importance, 1. without loss of information and 2. using the least amount of tokens possible: \n\n"

# account for the initial prompt tokens and a buffer
token_buffer=0
hard_context_limit=200
context_token_limit = (hard_context_limit-token_buffer)

def append_to_json_file(file_path, new_objects):
    # Read the existing data from the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Append new objects to the existing data
    data.append(new_objects)
    
    # Write the updated data back to the JSON file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def combine_chunks(chunks, extracted_text):

    #current_chunk = ""

    # Get the number of tokens of the extracted text
    num_tokens = chat.get_num_tokens(extracted_text)

    # If the number of tokens is less than the limit, return the chunks
    if num_tokens < context_token_limit:
        chunks.append(extracted_text.strip())
        return chunks

    # Get the length of the extracted text
    text_length = len(extracted_text)

    # Split the text in half (todo: to the closest sentence)
    text_chunk_1 = slice(0, text_length//2)
    text_chunk_2 = slice(text_length//2, text_length)

    chunks = combine_chunks(chunks, extracted_text[text_chunk_1])
    chunks = combine_chunks(chunks, extracted_text[text_chunk_2])

    #for word in words:
    #    if len(current_chunk) + len(word) <= word_count:
    #        current_chunk += word + ' '
    #    else:
    #        chunks.append(current_chunk.strip())
    #        current_chunk = word + ' '
    
    #if current_chunk:
    #    chunks.append(current_chunk.strip())

    return chunks


def split_pdf_into_chunks(file_path):
    chunks = []
    
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
            text = page.extract_text().replace('\n', ' ')  # Read the file and replace newlines with spaces
            page_num+=1
            # Append the extracted text to the result
            extracted_text += text

    chunks = combine_chunks(chunks, extracted_text)

    return chunks

def split_text_into_chunks(file_path):
    chunks = []
    
    with open(file_path, 'r') as file:
        extracted_text = file.read().replace('\n', ' ')  # Read the file and replace newlines with spaces

    chunks = combine_chunks(chunks, extracted_text)
    
    return chunks

def process_documents(source_folder, destination_folder):
    # Loop through the files in the source folder
    for filename in os.listdir(source_folder):

        if "Readme.md" != filename:

            # Get the full path of the current file
            source_file = os.path.join(source_folder, filename)

            # Check if the current item is a file
            if os.path.isfile(source_file):
                file_extension = os.path.splitext(source_file)[1].lower()
                if file_extension == '.pdf':
                    chunks = split_pdf_into_chunks(source_file)
                else:
                    chunks = split_text_into_chunks(source_file)
            
                for i in chunks:
                    # Get Full Embeddings for the Text Chunk
                    embeds = get_embedding(i,engine='text-embedding-ada-002')

                    obj = {
                        'uid': str(id(embeds)),
                        'embeddings': embeds,
                        'meta': {
                            'title': filename,
                            'file': filename,
                            'text': i  # Add the 'text' to the metadata
                        }
                    }

                    append_to_json_file('knowledge.json',obj)

                # Move the processed document to the destination folder
                destination_file = os.path.join(destination_folder, filename)
                shutil.move(source_file, destination_file)
                print(f"Moved {filename} to {destination_folder}")

# Process the Documents
process_documents('Learnin', 'Learnt')




