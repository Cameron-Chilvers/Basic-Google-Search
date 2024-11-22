from flask import Flask, render_template, request
import sys
import re
from time import time
from fuzzywuzzy import fuzz
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import numpy as np

sys.path.append('..')
from neo4j_connection import Neo4jConnection

app = Flask(__name__)

MAIN_TEMP = "main_page.html"
STOP_WORDS = set([
'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 
'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 
'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 
'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
])

# Main route
@app.route("/")
def main_page():
    return render_template(MAIN_TEMP)


def create_strict_text_query(search_phrase):
    words = search_phrase.split()
    word_conditions = ["(?i).*" + word + ".*" for word in words if word not in STOP_WORDS]
    query = "MATCH (n:Website) WHERE " + " AND ".join(["n.title =~ '{}'".format(condition) for condition in word_conditions]) + " RETURN n"

    return query

def create_expression(word):
    suffixes = ['ing', 'ly', 'ed', 'er', 'ion', 'ible', 'able', 'ment', 'ness', 'ist', 'ful', 'less', 'ous', 's']

    pattern = re.compile(r'(' + '|'.join(suffixes) + ')$')
    root = pattern.sub('', word)
    return '(?i).*' + root + '.*'

def create_similar_text_query(search_phrase):
    words = search_phrase.split()
    word_conditions = [create_expression(word) for word in words if word not in STOP_WORDS]
    query = "MATCH (n:Website) WHERE " + " AND ".join(["n.title =~ '{}'".format(condition) for condition in word_conditions]) + " RETURN n"

    return query

def create_vectorisation_model():
    # Creating the model for vectorisation
    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {"device": "cuda"}  # Ensure the model uses the GPU
    encode_kwargs = {"normalize_embeddings": True}
    hf_model = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)

    return hf_model


def create_vectorisation_query():
    query = """
    CALL db.index.vector.queryNodes('html_embeddings', 50, $embedding)
    YIELD node AS similarAbstract, score
    RETURN similarAbstract, score
    """

    return query

def preprocess(text):
    # Convert to lowercase and remove non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text.lower())
    words = text.split()
    # Remove stop words
    filtered_words = [word for word in words if word not in STOP_WORDS]
    return ' '.join(filtered_words)

def score_title(title, search_phrase):
    title = preprocess(title)
    search_phrase = preprocess(search_phrase)
    title_words = set(title.split())
    search_words = set(search_phrase.split())
    
    # Simple intersection over union as relevance score
    if not search_words:
        return 0  # Avoid division by zero if search phrase is empty
    relevance_score = len(title_words & search_words) / len(title_words | search_words)
    return relevance_score

def calculate_precision_recall(titles, search_phrase, relevant_titles, threshold=0.30):
    # Determine relevance based on scoring
    retrieved_relevant = [title['title'] for title in titles if score_title(title['title'], search_phrase) >= threshold]
    
    # Sets for precision and recall calculation
    retrieved_set = set(retrieved_relevant)
    relevant_set = set(relevant_titles)
    true_positives = len(retrieved_set & relevant_set)

    # Calculate precision and recall
    precision = true_positives / len(retrieved_set) if retrieved_set else 0
    recall = true_positives / len(relevant_set) if relevant_set else 0

    return precision, recall


# Using Levenshtein distance to find revlaant title names since we don't have user feedback
def find_relevant_titles_fuzzy(titles, search_phrase, threshold=40):
    relevant_titles = []
    normalized_search_phrase = search_phrase.lower()

    for title in titles:
        # The ratio function can be used for single word matching
        if fuzz.token_sort_ratio(normalized_search_phrase, title['title'].lower()) >= threshold:
            relevant_titles.append(title['title'])

    return relevant_titles

def cosine_similarity(embedding1, embedding2):
    # Ensure the embeddings are numpy arrays
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    
    # Calculate the dot product of the two embeddings
    dot_product = np.dot(embedding1, embedding2)
    
    # Calculate the norm (magnitude) of each embedding
    norm_embedding1 = np.linalg.norm(embedding1)
    norm_embedding2 = np.linalg.norm(embedding2)
    
    # Compute the cosine similarity
    cosine_sim = dot_product / (norm_embedding1 * norm_embedding2)
    
    return cosine_sim

def parse_db_results(results, use_embedding, search_embedding = None):
    result_list = []
    for results_ in results: # list of keys
        for data in results_:
            if use_embedding:
                results_[data]['score'] = results_['score']

            if search_embedding is not None:
                results_[data]['score'] = cosine_similarity(results_[data]['embedding'], search_embedding)

            result_list.append(results_[data])
        
            if use_embedding:
                # Only loop the once to get the data ignoring score
                break
    return result_list

# End point for the search phrase
@app.route("/send_input_phrase", methods=["GET","POST"])
def send_input_phrase():
    start = time()
    # Checking whats being printed
    print("All Data:", request.form)

    # Getting search phrase and checking if its there
    search_phrase = request.form.get("search_phrase")
    if search_phrase is None:
        return render_template(MAIN_TEMP, list__ = [{"title":"NO DATA FOUND"}])
    print("Search Phrase: ", search_phrase)

    # Getting if user wants to use embedding
    vector_ranking = request.form.get("embedding")
    if vector_ranking is None:
        print("No Vector Ranking")

    print("Vector Ranking? ", vector_ranking)

    # Getting Page Rank Sorting
    page_rank = request.form.get("page_rank")
    if page_rank is None:
        print("No Page Rank")
    print("Page Rank? ", page_rank)

    # Getting which search to use
    search_method = request.form.get('text_search')
    print("Search Method: ", search_method) 

    # Only creating hf_model if needed
    search_embedding = None
    if vector_ranking is not None or search_method == "vectorisation_search":
        hf_model = create_vectorisation_model()
        search_embedding = hf_model.embed_query(search_phrase)


    # Connecting to the database and querying it
    db_conn = Neo4jConnection("bolt://localhost:7687", "neo4j", "Password")
    query_parameters = None

    # Method to search for strict text match
    if search_method == "strict_text_search":
        query = create_strict_text_query(search_phrase)
        results = db_conn.execute_query(query, query_parameters)

        # Parsing Results
        result_list = parse_db_results(results, use_embedding=False, search_embedding=search_embedding)

    # Method to search for similar text
    elif search_method == "similar_text_search":
        # Setting up for the query
        query = create_similar_text_query(search_phrase)
       
        # Querying db
        results = db_conn.execute_query(query, query_parameters)
        
        # Parsing Results
        result_list = parse_db_results(results, use_embedding=False, search_embedding=search_embedding)

    # Method for Vector Search 
    elif search_method == "vectorisation_search":
        # Setting up for the query
        query_parameters = {'embedding': search_embedding}
        query =  create_vectorisation_query()

        # Querying db
        results = db_conn.execute_query(query, query_parameters)

        # Parsing Results
        result_list = parse_db_results(results, use_embedding=True)
            

    else: # No search method chosen
        db_conn.close()
        return render_template(MAIN_TEMP, list__ = [{"title":"NO DATA FOUND"}])

    # Closing db when done 
    db_conn.close()

    # Calculating precison and recall
    relevant_titles = find_relevant_titles_fuzzy(result_list, search_phrase)
    precision, recall = calculate_precision_recall(result_list, search_phrase, relevant_titles)

    # Creating the stats lists
    stats = [("Time Taken: ", time()-start), ("Number of Results: ", len(results)), ("Precision Score: ", precision), ("Recall Score: ", recall)]
    prev_search = [("Search Phrase: ", search_phrase), ("Search Method: ", search_method), ("Page Rank? ", "Yes" if page_rank else "No"), ("Embeddings? ", "Yes" if vector_ranking else "No")]

    # Telling user no results
    if len(result_list) == 0:
        result_list = [{'n':{"title":"No Results"} }]

    else:

        # Sorting the results to what the user wants
        if page_rank is not None and vector_ranking is not None: # weighting page rank so its not insiginficant
            result_list = sorted(result_list, key=lambda d: (d['page_rank']*100000) + d['score'], reverse=True)

        elif page_rank is not None:
            result_list = sorted(result_list, key=lambda d: d['page_rank'], reverse=True)

        elif vector_ranking is not None:
            result_list = sorted(result_list, key=lambda d: d['score'], reverse=True)


    return render_template(MAIN_TEMP, list__ = result_list, prev_search = prev_search, stats = stats)
