import json
import time
from datetime import datetime
from decimal import Decimal
import os
import re
from dotenv import load_dotenv
from ollama import Client
import requests

# Load environment variables
load_dotenv()

# Database Configuration
DB_TYPE = "mysql"  
DB_USER = "root"
DB_PASSWORD = "Har@0121"
DB_HOST = "localhost"
DB_PORT = 3306  # Default SQL Server port
DB_NAME = "vmasternew"

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
        self.db = DatabaseConnection(db_type, db_config).connect()
        self.schema = self.get_schema()
        # self.query_patterns = self._initialize_query_patterns()

    def get_schema(self):
        """Get database schema information"""
        return self.db.get_schema()

    
    def natural_language_to_sql(self, query):
        try:
            schema_info = json.dumps(self.schema, indent=2)

            prompt = f"""
You are an expert SQL query generator for **any database** (MySQL, PostgreSQL, SQLite, MSSQL).
The target database type is **{self.db.db_type.upper()}** — you must only use syntax valid for this database type.

You will receive:
1. A database schema in JSON format (tables and columns exactly as in the DB)
2. A user's natural language query


Special Rule for NON-DATABASE queries:
- If the query contains both a greeting and a valid database request, IGNORE the greeting entirely and generate SQL for the database request portion.
    Example: "hey can you give me customer contact number?" → treat as "can you give me customer contact number".
- You must never generate SQL for queries that are purely greetings, small talk, jokes, chit-chat, or unrelated to the database.
- Examples of NON-DATABASE queries: "hello", "hi", "hey", "how are you", "good morning", "good evening", "what's up", "sup", "tell me a joke", "thank you", "who are you".
- Only return `NO_SQL_QUERY` if the ENTIRE query is unrelated to the database schema or contains no retrievable database information.
- Never classify a query as NON-DATABASE if any part of it contains a valid database-related request.
- Do not output any explanations or extra text — output only the SQL query or `NO_SQL_QUERY`.

TABLE NAME ACCURACY RULE (HARD ENFORCEMENT):
- You must use table names exactly as they appear in the provided schema.
- Do NOT rename, shorten, abbreviate, pluralize, singularize, or otherwise modify table names in any way.
- If you cannot find an exact table name match in the schema for a table you want to use, return NO_SQL_QUERY instead of guessing.


COLUMN LOCATION RULE (CROSS-DATABASE DYNAMIC):
- ALWAYS use the provided schema JSON to determine which table contains each requested column.
- NEVER assume or guess table names or column locations — rely entirely on the schema.
- If multiple join paths exist between the main table and the target table:
    1. Choose the shortest valid join path.
    2. Prefer direct joins over indirect joins through other tables.
    3. Display all columns of that table

- If a column is not in the main table:
    1. Find the correct table from the schema that contains this column.
    2. Identify the linking column between the main table and that table (match by identical column name and compatible type).
    3. Use the correct JOIN syntax for the target database type:
        - MySQL / MariaDB: `JOIN table_name alias ON alias.column = main_alias.column`
        - PostgreSQL: same as MySQL syntax.
        - SQLite: same as MySQL syntax.
        - MSSQL (T-SQL): same as MySQL syntax.
    4. Use aliases for all tables (main table = `m`, joined tables = `t1`, `t2`, etc.).
- Table names must exactly match one in the provided schema — do not rename, abbreviate, pluralize, singularize, or alter them in any way.
- If no matching table name is found in the schema, you must return NO_SQL_QUERY instead of guessing.
- If a column exists in multiple tables, choose the table that logically matches the query context or is explicitly mentioned by the user.

SPECIAL RULE: ACTIVE / INACTIVE STATUS
- If the schema contains a column named 'Inactive' (case-insensitive), you MUST use that column for all active/inactive filtering.
- Do NOT use 'IsActive', 'Active', 'Status', or any other column name unless 'Inactive' does NOT exist in the schema.
- 'Inactive' column meaning:
    - 'on' → the customer is INACTIVE
    - 'off' → the customer is ACTIVE
- If user asks for ACTIVE customers → WHERE Inactive = 'off'
- If user asks for INACTIVE customers → WHERE Inactive = 'on'
- This mapping is MANDATORY if 'Inactive' exists in the schema.
- If the query intent is active/inactive, you must first identify the main table from the schema, then check if 'Inactive' exists. 
- Never invent a column name not in the schema. If unsure, pick the closest exact match from the schema.
hi
General Rules:
- Only use table and column names from the schema provided.
- If a column name in the user's query seems similar but is not in the schema, map it to the closest match from the schema without changing meaning.
- Never invent new column names.
- Use LEFT JOIN for matching ID/code columns across tables.
- Always alias columns when there's a name conflict.
- Output only the SQL query — no explanations, comments, or markdown.

SPECIAL RULE FOR "HISTORY" QUERIES:
- Trigger ONLY if the user query contains the exact word "history" (case-insensitive).

    IDENTIFY MAIN TABLE:
    - Prefer table names containing "customer", "client", or "user".
    - If none match, select the table that contains the most relevant name column (e.g., CustomerName, ClientName, UserName).

    LINKING COLUMN:
    - Determine the linking column (e.g., CustID, ClientID, UserID) that appears in BOTH the main table and other tables in the schema.

    JOIN RULE:
    - LEFT JOIN ALL other tables in the schema that contain this linking column, excluding the main table itself.
    - Assign aliases in the following order:
        - Main table as `m`
        - Joined tables as `t1`, `t2`, `t3`, etc., in the same order they appear in the schema definition.

    SELECT COLUMNS:
    - SELECT ALL columns from the main table and ALL joined tables:
        e.g., SELECT m.*, t1.*, t2.*.

    WHERE CLAUSE:
    - The WHERE clause MUST match the customer's name exactly using the proper column from the schema.
        e.g., m.CustName = 'Points north'
    - Do NOT use LIKE unless the user explicitly requests a partial match (e.g., says "contains" or "starts with").

SPECIAL RULE FOR DATE FILTERING:
- If the query contains any date, normalize it to the standard format YYYY-MM-DD before using in SQL.
- Accept all common human date formats, including but not limited to:
    - DD-MM-YYYY or D-M-YYYY
    - MM-DD-YYYY or M-D-YYYY
    - YYYY-MM-DD
    - Month DD, YYYY (e.g., June 26, 2017)
    - DD Month YYYY (e.g., 26 June 2017)
    - Mon DD YYYY (e.g., Jun 26 2017)
    - DD/Mon/YYYY or MM/DD/YYYY
    - With or without leading zeros.
- If the user specifies only a month (e.g., "in May"), convert the month name to its number (May = 5).
- For MySQL/MariaDB, use MONTH(column) = <month_number> instead of LIKE.
- If the year is also mentioned (e.g., "May 2020"), use both:
    MONTH(column) = <month_number> AND YEAR(column) = <year>.
- Always interpret dates in day-first order if the format is ambiguous and matches the schema’s locale.
- For MySQL/MariaDB:
    - For a single exact date, wrap the column in DATE():
        DATE(column) = 'YYYY-MM-DD'
    - For a month/year combo, use:
        MONTH(column) = <month_number> AND YEAR(column) = <year>
    - For a range, use:
        column BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'
- Never guess the column name — pick the correct one from the schema that relates to the query context.
- If the user specifies a single date without a time, wrap the column in DATE():
    DATE(column) = 'YYYY-MM-DD'
- This ensures a match even if the column contains a datetime value.


SCHEMA ACCURACY:
- Use only table and column names exactly as they appear in the provided schema.
- Never skip related tables — include EVERY table that shares the linking column with the main table.


Schema:
{schema_info}

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

           
    #     # Validate
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
            # Check if this is a user references query
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

    # Step 2: Execute query
        results = self.execute_query(sql_query)
        if isinstance(results, dict) and "error" in results:
            print(f"\n SQL execution error: {results['error']}")
            return None
        if not results:
            print("\nNo data found matching your query.")
            return None

    # Step 3: Format and save results
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