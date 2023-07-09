
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity

import openai
import json
import numpy
import tiktoken

from langchain.schema import AIMessage,HumanMessage,SystemMessage

general_model = "gpt-3.5-turbo"
embedding_model = "text-embedding-ada-002"

token_buffer=500
hard_context_limit=4097
context_token_limit = (hard_context_limit-token_buffer)

chat = ChatOpenAI(temperature=0,model=general_model)

messages = [
    SystemMessage(content="You are a helpful assistant"),
]

def get_context(inputPrompt, embeddings):
    search_term_vector = embeddings
    
    with open("knowledge.json",encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
        for item in data:
            item['embeddings'] = numpy.array(item['embeddings'])

        for item in data:
            item['similarities'] = cosine_similarity(item['embeddings'], search_term_vector)

        sorted_data = sorted(data, key=lambda x: x['similarities'], reverse=True)
        
        context = ''
        for i in sorted_data:
            if i['similarities'] > 0.6:
                encoding_name = tiktoken.encoding_for_model(general_model)
                tokenizer = tiktoken.get_encoding(encoding_name.name)
                num_tokens = len(tokenizer.encode(context+i['meta']['text']))
                if num_tokens < context_token_limit:
                    context += i['meta']['text'] + '\n'
            
    return context

user_input = input("you: ")

while user_input != "exit":
    context = ''
    embds = get_embedding(user_input, engine='text-embedding-ada-002')
    context = get_context(user_input, embds)
    
    messages.append(HumanMessage(content="context: \n" +context+"\n\nplease answer the following question using the above given context\n\n"+user_input))
    ai_response = chat(messages=messages).content
    messages.pop()
    print("ai: ",ai_response)
    #messages.append(AIMessage(content=ai_response))
    user_input = input("you: ")

