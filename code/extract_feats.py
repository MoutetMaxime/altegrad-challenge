import random
import re

from prompt_encoding import PromptEncoder

random.seed(32)


def extract_numbers(text):
    # Use regular expression to find integers and floats
    numbers = re.findall(r'\d+\.\d+|\d+', text)
    # Convert the extracted numbers to float
    return [float(num) for num in numbers]


def extract_feats(file):
    stats = []
    fread = open(file,"r")
    line = fread.read()
    line = line.strip()
    stats = extract_numbers(line)
    fread.close()
    return stats


def extract_encoding(text: str, prompt_encoder: PromptEncoder, tokenizer):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    sentence_embeddings = prompt_encoder(encoded_input)
    return sentence_embeddings


def extract_prompt_encodings(file: str, prompt_encoder: PromptEncoder, tokenizer):
    fread = open(file, "r")
    line = fread.read()
    line = line.strip()
    return extract_encoding(line, prompt_encoder, tokenizer)
