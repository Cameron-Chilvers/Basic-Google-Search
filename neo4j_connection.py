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

    def add_dict(self, dict_data):
        with self.__driver.session() as session:
            result = session.write_transaction(self.__create_and_return_node, dict_data)
            #print("Node created with labels and properties:", result)

    @staticmethod
    def __create_and_return_node(tx, dict_data):
        query = (
            "MERGE (n:Website {id: $id, title: $title, html: $html}) "
            "RETURN n"
        )
        result = tx.run(query, id=dict_data['file_name'], title=dict_data['title'], html=dict_data['html'])
        try:
            return [{"n": record["n"].get("properties")} for record in result]
        except Exception as e:
            print("Error creating node:", e)
            return None
        
    def add_batch(self, batch):
        with self.__driver.session() as session:
            session.write_transaction(self.__create_and_return_nodes_batch, batch)

    @staticmethod
    def __create_and_return_nodes_batch(tx, batch):
        for record in batch:
            query = "MERGE (n:Website {id: $id, title: $title, html: $html}) RETURN n"
            tx.run(query, id=record['file_name'], title=record['title'], html=record['html'])


    def add_batch_prop(self, batch):
         with self.__driver.session() as session:
            session.write_transaction(self.__add_prop_nodes_batch, batch)

    @staticmethod
    def __add_prop_nodes_batch(tx, batch):
        query = """
            UNWIND $batch AS line
            MATCH (n:Website {id: line[0]})
            SET n.link_address = line[1]
            RETURN n
            """
        tx.run(query, batch=batch)

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