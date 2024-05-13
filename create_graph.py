from Node_File import Node, NodeMap
from Graph import Graph
import glob
import json
import os
import re
import concurrent.futures
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from time import time
import dask.dataframe as dd
import ijson
from neo4j_connection import Neo4jConnection

NODE_PATH = "json_dumps\\nodes\\"

HTML_PATH = "json_dumps\\html\\"
HTML_PATH_HTMLS = r"D:\UNi\UTS\network\NetworkAssigment3\html\htmls-001.tar\htmls-001\htmls\\"
FAILED_LOG_PATH = os.path.join(HTML_PATH, "failed_files.log")

DATABASE_PATH = "json_dumps\\database\\"

# loop through links to create the grpah
def create_graph():
    graph = Graph()

    with open(r"nodes-links\links.txt", "r") as f:
        links = f.readlines()

    link_tuples = []

    for val in links:
        # Cleaning the values
        val = val.strip()
        link_split = val.split(' ')

        # Converting to int
        link_split[0] = int(link_split[0])
        link_split[1] = int(link_split[1])

        val_tuple = (link_split[0], link_split[1])

        # Adding to the graph
        graph.add_to_graph(link_split[0])
        graph.add_to_graph(link_split[1])

        # Adding the link
        graph.add_edge(link_split[0], link_split[1])

        #print(graph.graph_matrix)
        link_tuples.append(val_tuple)

    graph.dump_graph(r"json_dumps\graph\graph")

    return link_tuples

# Loading grpah from the json file
def load_graph():
    graph = Graph()

    graph.load_graph_from_json(r"json_dumps\graph\graph.json")

    return graph

# Loop through hosts to create the nodes
def create_nodes():
    with open(r"nodes-links\hosts.txt", 'r') as f:
        nodes = f.readlines()

    #print(nodes[5:])
    node_list = []
    for node in nodes:
        # Cleaning the values
        node = node.strip()

        node_split = node.split(" ")

        node_split[0] = int(node_split[0])
        node_split[1] = node_split[1]

        node = Node()
        node.initalise_node(node_split[0], node_split[1])

        node_list.append(node)
        #node.dump_node(NODE_PATH)

    return node_list

def dump_nodes(node_list):
    node_dict = dict()
    for node in node_list:
        node_prop = node.create_prop_dict()

        node_dict[node_prop["id"]] = node_prop

    with open(NODE_PATH + "nodes.json", "w") as f:
        json.dump(node_dict, f, indent=5)

# BROKEN
def read_nodes():
    for file in glob.glob(NODE_PATH + "*.json"):
        node = Node()

        node.load_from_json(file)

        print(node.print_node())


######################### HTML Parsing #########################
def process_html_file(file_path,):
    file_name = os.path.splitext(os.path.basename(file_path))[0]  # Strip extension
    try:
        with open(file_path, 'r', encoding="utf8", errors='replace') as f:
            html_string = f.read()
            match = re.search(r'<title>(.*?)</title>', html_string, re.IGNORECASE)
            title = match.group(1) if match else "Untitled"

            return {
                "file_name": file_name,
                "title": title,
                "html": html_string
            }
    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None

def save_batch_to_json(batch, batch_index):
    batch_path = os.path.join(HTML_PATH, f"html_batch_{batch_index:04d}.json")
    batch_dict = {item['file_name']: item for item in batch if item}

    with open(batch_path, 'w', encoding='utf-8') as f:
        json.dump(batch_dict, f, indent=5)  # Save only successful entries

def add_html_to_node(batch_size=500):
    file_paths = glob.glob(HTML_PATH_HTMLS + "*.html")
    num_batches = len(file_paths) // batch_size + (1 if len(file_paths) % batch_size else 0)

    # Process files in batches and handle exceptions
    for i in range(num_batches):
        file_paths_batch = file_paths[i * batch_size:(i + 1) * batch_size]
        batch_results = []
        with ProcessPoolExecutor() as executor:
            future_to_file = {executor.submit(process_html_file, fp): fp for fp in file_paths_batch}
            for future in concurrent.futures.as_completed(future_to_file):
                result = future.result()
                if result:
                    batch_results.append(result)
                else:
                    failed_path = future_to_file[future]
                    print(f"Failed processing: {failed_path}")

        save_batch_to_json(batch_results, i)

    print(f"All files extracted and saved in batches of {batch_size}")
    print(f"Failed files logged to {FAILED_LOG_PATH}")
######################### HTML Parsing #########################



######################### Single Search #########################
def find_word_in_files(search_word, file_path):
    start = time()
    results = []
    for index, file in enumerate(glob.glob(file_path + "*.json")):
        try:
            with open(file, 'rb') as f:  # ijson requires binary mode
                # Assuming your JSON structure has an array of objects at the root
                items = ijson.kvitems(f, "")  # Adjust 'item' path depending on JSON structure
                
                for key, item in items:
                    title = item['title'].lower()  # Access the title and handle cases
                    if any(x.lower() in title for x in search_word):  # Case insensitive search
                        results.append(item)

        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")

        #print(dataframe.head())
        
    print(len(results))
    
    print("time taken " + str(time() - start))
######################### Single Search #########################


######################### Single Search DASK #########################
def find_word_sinlge_dask(directory):
    for index, file in enumerate(glob.glob(directory + "*.json")):
        print(file)
        try:
            dask_frame = dd.read_json(file, lines = True)

            print(dask_frame.describe())    
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")

        #print(dataframe.head())
######################### Single Search DASK #########################


######################### Search Funcitons #########################
def search_in_file(file, search_word):
    try:
        dataframe = pd.read_json(file).T
        contain_values = dataframe[dataframe["title"].str.contains(search_word, case=False, na=False)]
        if not contain_values.empty:
            contain_values['file'] = file  # Add filename to results for reference
            return contain_values
    except Exception as e:
        print(f"Error processing file {file}: {str(e)}")
    return pd.DataFrame()  # Return empty DataFrame on failure or no match

def find_word_in_files_multi(search_word, filepath, num_threads=4):
    start = time()
    files = glob.glob(filepath + "*.json")
    results = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(search_in_file, file, search_word) for file in files]
        for future in futures:
            result = future.result()
            if not result.empty:
                results.append(result)

    # Concatenate all DataFrames into a single DataFrame
    if results:
        final_results = pd.concat(results, ignore_index=True)
        print(f"Found {len(final_results)} matches.")
    else:
        final_results = pd.DataFrame()
        print("No matches found.")

    print("Time taken: {:.2f} seconds".format(time() - start))
    return final_results
######################### Search Funcitons #########################


######################### Combine Funcitons #########################
def read_json(file_path):
    # Read a JSON file into a DataFrame
    with open(file_path, 'r') as f:
        file = json.load(f)

    return file

def combine_json_files(directory_path, output_path):
    # List all JSON files in the directory
    json_files = glob.glob(os.path.join(directory_path, '*.json'))
    
    # Getting Pairs
    for i in range(0, len(json_files), 2):
        pair = json_files[i:i+2]
        
        pair_one_name = os.path.basename(pair[0]).replace(".json", "").replace("html_batch_", "")
        try:
            pair_two_name = os.path.basename(pair[1]).replace(".json", "").replace("html_batch_", "")

            # Combine all DataFrames into a single DataFrame
            json_dict = {**read_json(pair[0]), **read_json(pair[1])}
       
        except:
            pair_two_name = ""
            json_dict = read_json(pair[0])

        file_path = os.path.join(output_path, pair_one_name + "_" + pair_two_name + ".json")

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_dict, f, indent=5)  # Save only successful entries

    print(f"Combined JSON saved to {output_path}")
######################### Combine Funcitons #########################


######################### Convert to line del Funcitons #########################
def convert_to_line_del(filepath):
    start = time()
    files = glob.glob(filepath + "*.json")

    output_dir = "json_dumps\\line_del\\"

    for file in files:
        file_name = os.path.splitext(os.path.basename(file))[0].replace("html_batch_", "")
        output_path = output_dir + file_name + ".json"
        
        # So ik where up to
        print(file_name)

        with open(file, 'r') as f:
            file = json.load(f)

        #print(file)
        with open(output_path, 'w') as f:
            for key, val in file.items():
                json_line = json.dumps(val)
                f.write(json_line + '\n')
        break
    print("time taken " + str(time() - start))
######################### Convert to line del Funcitons #########################

######################### ADD NEO4j Funcitons #########################
def add_to_neo4j(filepath, conn):
    start = time()
    files = glob.glob(filepath + "*.json")

    for file in files:
        file_name = os.path.splitext(os.path.basename(file))[0].replace("html_batch_", "")
        
        # So ik where up to
        print(file_name)

        with open(file, 'r') as f:
            file = json.load(f)

        #print(file)
        for key, val in file.items():
            failed_ids = []
            try:
                conn.add_dict(val)
            except:
                failed_ids.append(val["file_name"])

        if len(failed_ids) > 0:
            with open(r"failed.txt", 'a') as f:
                f.write("\n\n FAILED FOR FILE: " + file_name)
                f.writelines(failed_ids)
    print("time taken " + str(time() - start))
######################### ADD NEO4j Funcitons #########################


######################### ADD NEO4j BATCH Funcitons #########################
def add_to_neo4j_batch(filepath, conn, batch_size=100):
    start = time()
    files = glob.glob(filepath + "*.json")
    failed_ids = []

    for file_path in files:
        file_name = os.path.splitext(os.path.basename(file_path))[0].replace("html_batch_", "")
        
        print(f"Processing {file_name}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        # Prepare batches of records to send to Neo4j in a single transaction
        batch = []
        for key, val in data.items():
            batch.append(val)
            if len(batch) >= batch_size:
                try:
                    conn.add_batch(batch)
                except Exception as e:
                    print(f"Failed to insert batch from file {file_name}: {e}")
                    failed_ids.extend([d['file_name'] for d in batch])
                batch = []

        # Insert any remaining records in the last batch
        if batch:
            try:
                conn.add_batch(batch)
            except Exception as e:
                print(f"Failed to insert final batch from file {file_name}: {e}")
                failed_ids.extend([d['file_name'] for d in batch])

    if failed_ids:
        with open(r"failed.txt", 'a') as f:
            f.write("\n\nFAILED FOR FILE: " + file_name + "\n")
            f.writelines(failed_ids)

    print("Total time taken: {:.2f} seconds".format(time() - start))
######################### ADD NEO4j BATCH Funcitons #########################



####################### MAIN FUNCTION BELOW ################################
if __name__ == "__main__":
    # Creaing graph and returning the edges in a list of tuples
    #print(create_graph())

    # Loading Grpah from the json file created above
    # graph = load_graph()
    # print(graph.get_out_edges("0"))

    # Creating list of the nodes and the link addresses from txt file
    #node_list = create_nodes()

    # Dumping the node list to a json file
    #dump_nodes(node_list)

    # BROKEN
    #read_nodes()

    # Reading in the html files and saving them in batches in json files
    #add_html_to_node(batch_size=1000)

    # Concatinate json


    #check_if_not_in_dict()

    # Combining the json files
    # json_dataframe = read_json_with_dask(HTML_PATH)
    # print(json_dataframe.head())

    # json_path = os.path.join(DATABASE_PATH, "html_title.json")
    # json_dataframe.to_json(json_path, orient='records', lines=True)

    find_word_sinlge_dask(r"json_dumps\line_del\\")
    #df_array_of_objects = dd.read_json(r'D:\UNi\UTS\network\NetworkAssigment3\json_dumps\page_rank\pagerank_results_with_details.json')

    #print(df_array_of_objects.head())

    #convert_to_line_del(r"json_dumps\html\\")

    # NEO4j Stuff
    # uri = "bolt://localhost:7687"
    # user = "neo4j"
    # password = "Password"
    # conn = Neo4jConnection(uri, user, password)

    # add_to_neo4j_batch(r"json_dumps\html\\", conn)

    # #add_to_neo4j(r"json_dumps\html\\", conn)
    # conn.close()
    #find_word_in_files(["beach"], "json_dumps\\pairs\\")

    #find_word_in_files_multi("beach | ball", filepath="json_dumps\\html\\", num_threads=8)

    # Creating the combined files
    # file_paths = [r"json_dumps\html", r"json_dumps\pairs", r"json_dumps\quads", r"json_dumps\octo", r"json_dumps\sixteen", r"json_dumps\thirtytwo"]
    # for pair in pairwise(file_paths):
    #     print(pair)

    #     combine_json_files(pair[0], pair[1])


    # Cpher to use for search

#     MATCH (n:Website)
#     WHERE n.title =~ '(?i).*beach.*'
#     RETURN n
