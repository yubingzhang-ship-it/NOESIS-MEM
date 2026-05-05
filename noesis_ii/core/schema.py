import sqlite3

class NoesisDB:
    def __init__(self, db_name='noesis.db'):
        self.connection = sqlite3.connect(db_name)
        self.create_tables()

    def create_tables(self):
        self.create_working_memory_table()
        self.create_ltm_nodes_table()
        self.create_ltm_links_table()
        self.create_memory_traces_table()

    def create_working_memory_table(self):
        query = '''CREATE TABLE IF NOT EXISTS working_memory (
            id INTEGER PRIMARY KEY,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );'''  
        self.execute_query(query)

    def create_ltm_nodes_table(self):
        query = '''CREATE TABLE IF NOT EXISTS ltm_nodes (
            id INTEGER PRIMARY KEY,
            node_data TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );'''  
        self.execute_query(query)

    def create_ltm_links_table(self):
        query = '''CREATE TABLE IF NOT EXISTS ltm_links (
            id INTEGER PRIMARY KEY,
            source_node INTEGER,
            target_node INTEGER,
            link_type TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_node) REFERENCES ltm_nodes (id),
            FOREIGN KEY (target_node) REFERENCES ltm_nodes (id)
        );'''  
        self.execute_query(query)

    def create_memory_traces_table(self):
        query = '''CREATE TABLE IF NOT EXISTS memory_traces (
            id INTEGER PRIMARY KEY,
            trace_data TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );'''  
        self.execute_query(query)

    def execute_query(self, query):
        cursor = self.connection.cursor()
        cursor.execute(query)
        self.connection.commit()
        cursor.close()

    def __del__(self):
        self.connection.close()

if __name__ == '__main__':
    db = NoesisDB()