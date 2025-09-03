import json
import time
from datetime import datetime
from decimal import Decimal
import os
import re
from dotenv import load_dotenv
from ollama import Client
import requests
from sentence_transformers import SentenceTransformer
import chromadb
import sqlparse
from sql_metadata import Parser


# Load environment variables
load_dotenv()

# Database Configuration
DB_TYPE = os.getenv("DB_TYPE", "mysql")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME", "")

# Ollama Configuration
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:4b"  # Using smaller model for faster processing
OLLAMA_TIMEOUT = 500  # 500 seconds timeout

# Initialize Ollama client with timeout
client = Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class DatabaseConnection:
    def __init__(self, db_type, config):
        self.db_type = db_type.lower()
        self.config = config
        self.connection = None
        self.cursor = None
        
    def connect(self):
        try:
            if self.db_type == 'mysql':
                import pymysql
                self.connection = pymysql.connect(**self.config)
            elif self.db_type == 'postgresql':
                import psycopg2
                self.connection = psycopg2.connect(**self.config)
            elif self.db_type == 'sqlite':
                import sqlite3
                self.connection = sqlite3.connect(self.config['database'])
            elif self.db_type == 'mssql':  # Added T-SQL support
                import pyodbc
                conn_str = (
                    f"DRIVER={{SQL Server}};"
                    f"SERVER={self.config['host']},{self.config['port']};"
                    f"DATABASE={self.config['database']};"
                    f"UID={self.config['user']};"
                    f"PWD={self.config['password']}"
                )
                self.connection = pyodbc.connect(conn_str)
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
            
            self.cursor = self.connection.cursor()
            return self
        except Exception as e:
            print(f"Database connection error: {str(e)}")
            raise
    
    def execute(self, query, params=None):
        """Execute SQL query with optional parameters"""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            return self.cursor
        except Exception as e:
            print(f"Query execution error: {str(e)}")
            raise
    
    def fetchall(self):
        try:
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Fetch error: {str(e)}")
            raise
    
    def close(self):
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
        except Exception as e:
            print(f"Error closing connection: {str(e)}")

    def get_schema(self):
        """Get database schema information"""
        schema = {}
        try:
            if self.db_type == 'mysql':
                # Get all tables
                self.execute("SHOW TABLES")
                tables = self.fetchall()
                
                for table in tables:
                    table_name = table[0]
                    self.execute(f"SHOW COLUMNS FROM {table_name}")
                    columns = self.fetchall()
                    schema[table_name] = [col[0] for col in columns]
                    
            elif self.db_type == 'postgresql':
                # Get all tables
                self.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                tables = self.fetchall()
                
                for table in tables:
                    table_name = table[0]
                    self.execute(f"""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = %s
                    """, (table_name,))
                    columns = self.fetchall()
                    schema[table_name] = [col[0] for col in columns]
                    
            elif self.db_type == 'sqlite':
                # Get all tables
                self.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = self.fetchall()
                
                for table in tables:
                    table_name = table[0]
                    self.execute(f"PRAGMA table_info({table_name})")
                    columns = self.fetchall()
                    schema[table_name] = [col[1] for col in columns]
                    
            elif self.db_type == 'mssql':  # Added T-SQL schema retrieval
                # Get all tables
                self.execute("""
                    SELECT TABLE_NAME 
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_TYPE = 'BASE TABLE'
                """)
                tables = self.fetchall()
                
                for table in tables:
                    table_name = table[0]
                    self.execute(f"""
                        SELECT COLUMN_NAME 
                        FROM INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_NAME = ?
                    """, (table_name,))
                    columns = self.fetchall()
                    schema[table_name] = [col[0] for col in columns]
                    
            return schema
        except Exception as e:
            print(f"Error getting schema: {str(e)}")
            raise

        
class QueryConverter:       
    def __init__(self, db_type, db_config):
        self.db_type = db_type
        self.db_config = db_config
        try:
            self.db = DatabaseConnection(db_type, db_config).connect()
            self.schema = self.get_schema()
        except Exception as e:
            raise Exception(f"Database connection failed: {str(e)}")
        
        if self.schema:
            self._init_rag()

    def _safe_metadata(self, value):
        """Convert any Python object into a ChromaDB-safe metadata value."""
        import json
        from datetime import datetime
        from decimal import Decimal

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (datetime, Decimal)):
            return str(value)
        # Lists, dicts, sets, tuples → JSON string
        return json.dumps(value, default=str)

    def _init_rag(self):
        # === RAG Setup (Embedding + Vector Store) ===
        try:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.chroma_client = chromadb.Client()
            self.collection = self.chroma_client.get_or_create_collection(name="db_schema")

            # Clear existing data (optional, so we refresh every run)
            try:
                existing_ids = self.collection.get()["ids"]
                if existing_ids:
                    self.collection.delete(ids=existing_ids)
            except Exception:
                pass

            # Prepare schema text for embeddings
            docs = []
            ids = []
            for table, columns in self.schema.items():
                content = f"Table: {table} | Columns: {', '.join(columns)}"
                docs.append(content)
                ids.append(table)

            embeddings = self.embedding_model.encode(docs).tolist()
            self.collection.add(documents=docs, embeddings=embeddings, ids=ids)
            print(f"[RAG] Stored {len(docs)} tables into Chroma vector store.")
        except Exception as e:
            # Fail open: continue without RAG
            print(f"Warning: RAG initialization failed: {e}")
            self.embedding_model = None
            self.chroma_client = None
            self.collection = None

    def get_schema(self):
        """Get database schema information"""
        return self.db.get_schema()

    def get_full_data(self):
        """Fetch ALL rows from each table (only safe for small DBs)."""
        full_data = {}
        try:
            for table in self.schema.keys():
                try:
                    self.db.execute(f"SELECT * FROM {table}")
                    rows = self.db.fetchall()
                    columns = [desc[0] for desc in self.db.cursor.description]
                    full_rows = [dict(zip(columns, row)) for row in rows]
                    full_data[table] = full_rows
                except Exception as e:
                    full_data[table] = [f"Error fetching data: {e}"]
        except Exception as e:
            print(f"Full data extraction failed: {e}")
        return full_data

    
    def natural_language_to_sql(self, query):
        try:
            # RAG Search: Get relevant schema chunks (if available)
            retrieved_docs = None
            if getattr(self, 'collection', None):
                try:
                    rag_results = self.collection.query(
                        query_texts=[query],
                        n_results=5
                    )
                    retrieved_docs = rag_results['documents'][0]
                except Exception as e:
                    print(f"Warning: RAG search failed: {e}")

            # Use only relevant parts of schema in the prompt
            schema_info = json.dumps(self.schema, indent=2)
            full_data = json.dumps(self.get_full_data(), indent=2)
            prompt = f"""
You are an expert SQL query generator for **any database** (MySQL, PostgreSQL, SQLite, MSSQL).
The target database type is **{self.db.db_type.upper()}** — you must only use syntax valid for this database type.

You will receive:
1. A database schema in JSON format (tables, columns and data exactly as in the DB)
2. A user's natural language query


Schema:
{schema_info}

Full Data:
{full_data}

User Request:
{query}

Relevant Schema (retrieved via semantic search):
{schema_info}

Only use these tables and columns. If something is missing here, assume it does not exist in the database.
User Query:
{query}

Output:
[Only return the SQL SELECT query, no other text]
"""

            print(f" Prompt sent to Ollama model: {OLLAMA_MODEL}")
            response = client.generate(
                model=OLLAMA_MODEL,
                prompt=prompt,
                stream=False,
                options={"seed": int(time.time())},  # Random seed for each run
                system="You are a SQL generator. Return only the SQL query."
            )
            sql_query = response.get("response", "").strip()
            if self.db.db_type == "mysql":
                sql_query = re.sub(
                r"CAST\s*\(\s*SUBSTR\s*\(\s*(\w+),\s*1,\s*4\s*\)\s*AS\s*INT\s*\)\s*=\s*(\d{4})",
                r"YEAR(\1) = \2",
                sql_query,
                flags=re.IGNORECASE
            )

        # Cleanup unwanted formatting
            sql_query = re.sub(r"```sql|```|/\*|\*/", "", sql_query).strip()

           
         # Validate
            if not sql_query.lower().startswith("select") or "from" not in sql_query.lower():
                print(f" Invalid SQL returned: {sql_query}")
                return None

            print(f"Cleaned SQL: {sql_query}")
            return sql_query

        except Exception as e:
            print(f" Model error: {str(e)}")
            return None

    def format_value(self, value, data_type):
        """Format value based on its data type"""
        if value is None:
            return "null"
        if isinstance(value, Decimal):
            return float(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (int, float)):
            return value
        return str(value)

    def get_table_name(self, sql_query):
        """Extract table name from SQL query"""
        try:
            query = sql_query.lower()
            from_index = query.find('from')
            if from_index != -1:
                after_from = query[from_index + 4:].strip()
                table_name = after_from.split()[0].strip('`"[]')
                return table_name
            return "query"
        except:
            return "query"

    def execute_query(self, sql_query):
        """Efficiently execute SQL query for customers and addresses, but do NOT nest addresses in the output."""
        try:
            main_table = self.get_table_name(sql_query)
            print(f"DEBUG: Main table detected: {main_table}")
            self.db.execute(sql_query)
            columns = [desc[0] for desc in self.db.cursor.description]
            results = self.db.fetchall()
            print(f"DEBUG: Fetched {len(results)} rows from database.")
            formatted_results = []
            for row in results:
                row_dict = {}
                has_data = False
                for i, col in enumerate(columns):
                    value = self.format_value(row[i], None)
                    if value is not None and value != "":
                        row_dict[col] = value
                        has_data = True
                if has_data:
                    formatted_results.append(row_dict)
            print(f"DEBUG: Formatted {len(formatted_results)} main results.")
            # Do NOT nest addresses at all
            print(f"DEBUG: Ready to return {len(formatted_results)} results.")
            return formatted_results
        except Exception as e:
            error_msg = str(e)
            if "Unknown column" in error_msg:
                import re
                match = re.search(r"Unknown column '([^']+)'", error_msg)
                if match:
                    column = match.group(1)
                    return {"error": f"Column '{column}' does not exist in the database schema"}
            return {"error": error_msg}

    def save_to_json(self, data, sql_query, filename_prefix=None):
        """Save results to JSON file"""
        if filename_prefix is None:
            if "UNION ALL" in sql_query and "user_id" in sql_query:
                filename_prefix = "user_references"
            else:
                table_name = self.get_table_name(sql_query)
                filename_prefix = f"{table_name}_results"
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, cls=CustomJSONEncoder)
        return filename 

    def process_query(self, natural_language_query):
        """Fully model-based: converts NL to SQL using Ollama + executes + saves result."""
        sql_query = self.natural_language_to_sql(natural_language_query)
        print(f"\nGenerated SQL: {sql_query}")
        if not sql_query:
            print("\n Model did not return a SQL query.")
            return None
    
        results = self.execute_query(sql_query)
        if isinstance(results, dict) and "error" in results:
            print(f"\n SQL execution error: {results['error']}")
            return None
        if not results:
            print("\nNo data found matching your query.")
            return None

        formatted_results = []
        for row in results:
            if 'table_name' not in row:
                table_name = self.get_table_name(sql_query)
                new_row = {'table_name': table_name}
                new_row.update(row)
                formatted_results.append(new_row)
            else:
                formatted_results.append(row)

        filename = self.save_to_json(formatted_results, sql_query)
        print(f"\nResults have been saved to: {filename}")
        return formatted_results

    def close(self):
        """Close database connection"""
        self.db.close()

def main():
    # Use hardcoded database configuration
    db_type = DB_TYPE
    db_config = {
        'host': DB_HOST,
        'port': DB_PORT,
        'user': DB_USER,
        'password': DB_PASSWORD,
        'database': DB_NAME
    }
     
    converter = QueryConverter(db_type, db_config)
    
    print("\n=== SQL to JSON Reporter ===")
    
    try:
        query = input("\nEnter your query: ").strip()
        
        if not query:
            print("Please enter a valid query.")
            return
            
        print("\nProcessing your query...")
        results = converter.process_query(query)
        
        if results is None:
            print("\nNo results found.")
        else:
            print(f"\nQuery completed successfully!")
    
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        converter.close()

        def unload_model(model_name):
            try:
                requests.post(f"{OLLAMA_HOST}/api/unload", json={"model": model_name})
                print(f" Model '{model_name}' unloaded from memory.")
            except Exception as e:
                print(f"Error unloading model: {e}")

        unload_model(OLLAMA_MODEL)
        print("\nProgram completed.")

if __name__ == "__main__":
    main() 