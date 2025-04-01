from bs4 import BeautifulSoup
import json
import pandas as pd

# def clean_html(html_content):
#     # Parse the HTML content
#     soup = BeautifulSoup(html_content, 'html.parser')

#     # Remove script and style elements
#     for script in soup(["script", "style"]):
#         script.decompose()

#     # Get text from the cleaned soup
#     text = soup.get_text()

#     # Optionally, remove leading and trailing spaces on each line
#     lines = (line.strip() for line in text.splitlines())
#     # Break multi-headlines into a line each
#     chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
#     # Drop blank lines
#     text = '\n'.join(chunk for chunk in chunks if chunk)

#     return text

# # Example usage
# with open(r"json_dumps\html\html_batch_0000.json", 'r') as f:
#     data = json.load(f)

# cleaned_text = clean_html(data['0']['html'])
# print(cleaned_text)

df = pd.read_csv('export.csv')
df = df.apply(lambda s:s.str.replace('"', ""))


print(df.head())
print('59006' in df.values)

# for id in df['n.id']:
#     print(id)

# print(df.head())
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# model_name = "BAAI/bge-small-en-v1.5"
# #model_name = "BAAI/bge-m3"
# model_kwargs = {"device": "cpu"}
# encode_kwargs = {"normalize_embeddings": True} # This to use cosine to make it work with neo4j
# hf = HuggingFaceBgeEmbeddings(
#     model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
# )

# query_result = hf.embed_query(cleaned_text)

# print(query_result)