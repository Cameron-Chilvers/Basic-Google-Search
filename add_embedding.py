import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from time import time
import os
import glob
import json
import pandas as pd
import html
from bs4 import BeautifulSoup
from lxml import etree
import torch
from torch.nn.functional import cosine_similarity
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from neo4j import GraphDatabase

class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)

    def close(self):
        if self.__driver:
            self.__driver.close()

    def add_batch_embed_page(self, batch):
        with self.__driver.session() as session:
            session.write_transaction(self.__add_embed_page_batch, batch)

    @staticmethod
    def __add_embed_page_batch(tx, batch):
        query = """
            UNWIND $batch AS line
            MATCH (n:Website {id: line[0]})
            SET n.page_rank = line[1]
            SET n.embedding = line[2]
            SET n.cleaned_html = line[3]
            SET n.title = line[4]
            RETURN n
            """
        tx.run(query, batch=batch)
    
    def execute_query(self, query, parameters=None):
        with self.__driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

async def get_page_rank_df():
    async with aiofiles.open(r"json_dumps\page_rank\output_all_items.json", 'r') as f:
        data = await f.read()
    data = json.loads(data)
    df = pd.DataFrame(data).set_index("id").sort_index()
    return df

def get_page_rank_df_simple():
    with open(r"json_dumps\page_rank\output_all_items.json", 'r') as f:
        data = json.load(f)

    data = pd.read_json(r"json_dumps\page_rank\output_all_items.json", orient='records')

    data.set_index("id", inplace=True)
    data.sort_index(inplace=True)
    return data

def embedd_text(text, model: HuggingFaceBgeEmbeddings):
    return model.embed_query(text)

def clean_html(text):
    def is_xml(content):
        try:
            etree.fromstring(content)
            return True
        except (etree.XMLSyntaxError, ValueError):
            return False
    
    if is_xml(text):
        try:
            parser = etree.XMLParser(recover=True)
            root = etree.fromstring(text, parser=parser)
            cleaned_text = etree.tostring(root, pretty_print=True, encoding='unicode')
        except Exception as e:
            print(f"Failed to clean XML: {e}")
            cleaned_text = text
    else:
        try:
            soup = BeautifulSoup(text, 'html.parser')
            cleaned_text = soup.get_text(separator=' ')
        except Exception as e:
            print(f"Failed to clean HTML: {e}")
            cleaned_text = text
    
    return cleaned_text

def calculate_cosine_similarity(embedding1, embedding2):
    tensor1 = torch.tensor(embedding1)
    tensor2 = torch.tensor(embedding2)
    return cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0)).item()

def process_file(file_path, page_rank_df, model, conn: Neo4jConnection, batch_size, ids= None):
    file_name = os.path.splitext(os.path.basename(file_path))[0].replace("html_batch_", "")
    print(f"Processing {file_name}")

    with open(file_path, 'r') as f:
        data = json.load(f)

    batch = []
    for key, val in data.items():
        if ids is not None and val['file_name'] not in ids.values:
            continue
        
        try:
            page_rank_value = page_rank_df.loc[int(key), 'pagerank']
        except:
            page_rank_value = 0

        try:
            cleaned_title = html.unescape(val['title'])
        except:
            cleaned_title = val['title']

        try:    
            cleaned_html = clean_html(val['html']).encode('utf-16', 'surrogatepass').decode('utf-16', 'replace')
            embedding = embedd_text(cleaned_html, model)
        except:
            embedding = []

        batch.append((key, page_rank_value, embedding, cleaned_html, cleaned_title))

        if len(batch) >= batch_size:
            try:
                conn.add_batch_embed_page(batch)
            except Exception as e:
                print(f"Failed to insert batch from file {file_name}: {e}")
            batch = []

    if batch:
        try:
            conn.add_batch_embed_page(batch)
        except Exception as e:
            print(f"Failed to insert final batch from file {file_name}: {e}")

async def embedd_html_batch(filepath, conn, model, batch_size=100):
    start = time()
    files = glob.glob(filepath + "*.json")
    page_rank_df = await get_page_rank_df()

    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, process_file, file_path, page_rank_df, model, conn, batch_size)
            for file_path in files
        ]
        await asyncio.gather(*tasks)

    print("Total time taken: {:.2f} seconds".format(time() - start))



df = pd.read_csv('export.csv')

# Usage Example
uri = "bolt://localhost:7687"
user = "neo4j"
password = "Password"
conn = Neo4jConnection(uri, user, password)

model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {"device": "cuda"}  # Ensure the model uses the GPU
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

page_rank_df = get_page_rank_df_simple()
df = df.apply(lambda s:s.str.replace('"', ""))



process_file(r"json_dumps\html\html_batch_0191.json", page_rank_df, hf, conn, 100, df)

#asyncio.run(embedd_html_batch(r"json_dumps\html\\", conn, hf))
