import tiktoken

CHUNK_MODEL = "gpt-3.5-turbo"
CHUNK_SIZE = 200  # The target size of each text chunk in tokens
MIN_CHUNK_SIZE_CHARS = 350  # The minimum size of each text chunk in characters
MIN_CHUNK_LENGTH_TO_EMBED = 5  # Discard chunks shorter than this
MAX_NUM_CHUNKS = 10000  # The maximum number of chunks to generate from a text

# Get the chunk model
def get_chunk_model():
    return CHUNK_MODEL

# Get the chunk size
def get_chunk_size():
    return CHUNK_SIZE

# Get the Minimum chunk size is characters
def get_min_chunk_size_chars():
    return MIN_CHUNK_SIZE_CHARS

# Get the Minimum Length in Characters that we accept, anything less is discarded
def get_min_chunk_length():
    return MIN_CHUNK_LENGTH_TO_EMBED

# Get the Maximum Number Of Chunks to generate From Text
def get_max_num_chunks():
    return MAX_NUM_CHUNKS

def get_tokens(text):
    encoding_name = tiktoken.encoding_for_model(CHUNK_MODEL)
    tokenizer = tiktoken.get_encoding(encoding_name.name)
    # Return the Tokenized text and the tokenizer
    return (tokenizer.encode(text, disallowed_special=()), tokenizer)

# Split the text into chunks
def combine_chunks(chunks, extracted_text):

    tokens_tuple = get_tokens(extracted_text)
    tokens_list = list(tokens_tuple)
    tokens = tokens_list[0]
    tokenizer = tokens_list[1]
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

# Get all them Chunks
def get_chunks(text):
    # Return an empty list if the text is empty or whitespace
    if not text or text.isspace():
        return []
    # Combine The Chunks Together and return them
    return combine_chunks([], text)