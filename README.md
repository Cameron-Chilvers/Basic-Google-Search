# Simple Google Search Using a Graph Database

A simple google search engine was made to investigate "How do search engines work and how can they be improved?".

The question is explored by creating a neo4j graph database consisting of old website and their links with a simple front end to query the database to test the search engine. The PageRank algorithm has been run over the website to also help determine what is the most efficient search method, and all of the website names have been vectorised. From this three different search methods and four different sorting methods were compared to get our results. 

### Search methods:
- Strict Text Search
- Similar Text Search
- Vector Search

### Sorting methods:
- ID Sorting
- PageRank Sorting
- Vectorisation sorting
- PageRank + Vectorisation Sorting

### Data
The data used is html data of old webpages and how they link to one another. For this experiment we have used 237,618 unique website to construct our database around and the data can be found in the below drive.

https://drive.google.com/drive/u/1/folders/1_KD8XnFOoRLrg7jYIH5MWtQkzUzwD8JP

To use download "html.taz.gz", and place it a folder called "html" in the root directory and run the above python scripts

### Results
It was found that when the search phrase was more general the strict text matching searching performed better than the vector search. With the similar text search always returning more results than strict text search which is a given. Since the vector search is done on the embeddings it was able to return results in different languages, whilst the text search can only return results in the user’s language.
