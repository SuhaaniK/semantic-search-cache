import re

def clean_text(text):

    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = text.replace('\n', ' ')
    text = text.lower()

    return text