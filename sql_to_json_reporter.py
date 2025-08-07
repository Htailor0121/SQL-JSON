import json
import time
from datetime import datetime
from decimal import Decimal
import os
import re
from dotenv import load_dotenv
from ollama import Client

# Load environment variables
load_dotenv()

# Database Configuration
DB_TYPE = "mysql"  
DB_USER = "root"
DB_PASSWORD = "Har@0121"
DB_HOST = "localhost"
DB_PORT = 3306  # Default SQL Server port
DB_NAME = "VMaster7"

# Ollama Configuration
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:4b"  # Using smaller model for faster processing
OLLAMA_TIMEOUT = 500  # 120 seconds timeout

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
        self.query_patterns = self._initialize_query_patterns()

    def get_schema(self):
        """Get database schema information"""
        return self.db.get_schema()

    def _initialize_query_patterns(self):
        """Initialize dynamic query patterns based on schema"""
        patterns = {}
        
        # Dynamically create patterns for each table
        for table_name, columns in self.schema.items():
            # Determine table type based on name and columns
            table_type = self._analyze_table_type(table_name, columns)
            
            # Find potential date and value fields
            date_fields = [col for col in columns if any(date_term in col.lower() for date_term in ['date', 'created', 'modified', 'updated', 'time'])]
            value_fields = [col for col in columns if any(value_term in col.lower() for value_term in ['price', 'amount', 'total', 'rate', 'quantity', 'cost', 'value'])]
            
            # Find potential foreign keys
            foreign_keys = {}
            for col in columns:
                if col.endswith('_id') or col.endswith('ID'):
                    referenced_table = self._find_referenced_table(col, table_name)
                    if referenced_table:
                        foreign_keys[referenced_table] = f"{table_name}.{col} = {referenced_table}.{referenced_table.replace('s', '')}ID"
            
            # Create pattern for this table
            patterns[table_name] = {
                'tables': [table_name],
                'joins': list(foreign_keys.keys()),
                'conditions': columns,
                'date_field': date_fields[0] if date_fields else None,
                'value_field': value_fields[0] if value_fields else None,
                'join_conditions': foreign_keys,
                'type': table_type
            }
            
            # Add common patterns based on table type
            if table_type == 'transaction':
                patterns[table_name]['conditions'].extend(['status', 'amount', 'date'])
            elif table_type == 'user':
                patterns[table_name]['conditions'].extend(['name', 'email', 'status'])
            elif table_type == 'product':
                patterns[table_name]['conditions'].extend(['name', 'price', 'stock'])
            elif table_type == 'menu':
                patterns[table_name]['conditions'].extend(['name', 'category', 'is_active'])
        
        return patterns

    def _analyze_table_type(self, table_name, columns):
        """Analyze table name and columns to determine table type"""
        table_name_lower = table_name.lower()
        columns_lower = [col.lower() for col in columns]
        
        if any(term in table_name_lower for term in ['payment', 'transaction', 'order']):
            return 'transaction'
        elif any(term in table_name_lower for term in ['user', 'customer', 'client']):
            return 'user'
        elif any(term in table_name_lower for term in ['product', 'item', 'goods']):
            return 'product'
        elif any(term in table_name_lower for term in ['menu', 'category']):
            return 'menu'
        elif any(term in table_name_lower for term in ['log', 'history']):
            return 'log'
        elif any(term in table_name_lower for term in ['config', 'setting']):
            return 'config'
        else:
            return 'other'

    def _find_referenced_table(self, column_name, current_table):
        """Find the table referenced by a foreign key column"""
        # Remove 'ID' or '_id' suffix
        base_name = column_name.lower().replace('_id', '').replace('id', '')
        
        # Try to find matching table
        possible_tables = [t for t in self.schema.keys() if t.lower().startswith(base_name)]
        
        if possible_tables:
            return possible_tables[0]
        return None

    def natural_language_to_sql(self, query):
        try:
         # Use shortened schema (first 5 tables only)
            schema_info = json.dumps(self.schema, indent=2)  # Use full schema
            # print("\n=== DATABASE SCHEMA ===")
            # print(json.dumps(self.schema, indent=2))

            prompt = f"""
You are an expert SQL query generator for MySQL.

You will be given:
- A full MySQL database schema in JSON format
- A user’s natural language query

Your job is to generate a **single, valid SQL SELECT query** that fulfills the user's request.

General Rules:
1. Use only tables and columns listed in the provided schema.
2. Never hallucinate or guess column/table names.
3. Use only exact column names (case-sensitive where applicable).
4. Use LEFT JOINs to join related tables where a foreign key like `CustID`, `CrewID`, etc., exists.
5. Only include JOINs between tables that share common columns.
6. If the user specifies a name, ID, or email, add an appropriate WHERE clause.
7. Use DISTINCT only if the user asks for “unique”, “distinct”, or “different types”.
8. Only use LIMIT when the user explicitly asks for “top”, “first”, “recent”, or “latest”.
9. Never use subqueries with `=` that return multiple rows — prefer `IN (...)` or JOINs.
10. Return only **one SQL SELECT query** — no explanations, no comments, no markdown.

Important Rules:-
Important Rules:
- If the user query includes words like "history", "everything", "full info", or "details" about a customer (name or ID):
  → Do the following:
    1. From the `customers` table, find the matching customer using: 
       WHERE customers.CustName LIKE '<name>'
    2. Identify **all tables that contain the column `CustID`**, including:
       - customers
       - addresses
       - custcontact
       - orders
       - etc. (based on schema)
    3. Generate a SQL SELECT with LEFT JOIN on all those tables using the `CustID` key.
    4. Return **all columns** from every table, with aliases to avoid conflicts.
- This way, you show the full history of that customer across all relevant tables.
- Do not use any column unless it appears exactly in the schema.
- Do not use columns like `status`, `name`, or `email` unless they are explicitly in the table schema.
- Use meaningful table aliases to keep the query readable.
- Never return more than one SELECT query. Never include explanations.


=== DATABASE SCHEMA ===
{schema_info}

=== USER QUERY ===
{query}

=== OUTPUT ===
[Only one valid SQL SELECT query]
"""

            print(f" Prompt sent to Ollama model: {OLLAMA_MODEL}")
            response = client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            stream=False,
            system="You are a SQL generator. Return only the SQL query."
        )
            sql_query = response.get("response", "").strip()

            sql_query = re.sub(r"\\b(\\w+)\\.\\1\\.", r"\\1.", sql_query)

            sql_query = sql_query.replace("```sql", "").replace("```", "").replace("*/", "").strip()
            if "history of" in query.lower():
                match = re.search(r"history of\s+([a-zA-Z_]+)\s+(.+)", query, re.IGNORECASE)
                if match:
                    entity = match.group(1).strip().lower()         
                    name_value = match.group(2).strip()             

                    main_table = None
                    for table in self.schema:
                        if entity in table.lower():
                            main_table = table
                            break

                    if not main_table:
                        print(f"Could not identify main table for entity: {entity}")
                        return None

                    main_id = None
                    main_columns = self.schema[main_table]

                    for col in main_columns:
                        col_lower = col.lower()
                        if col_lower in [f"{entity}id", f"{entity}_id"]:
                            main_id = col
                            break
                        elif col_lower.endswith("id") and entity[:3] in col_lower:
                            main_id = col
                            break
                    if not main_id:
                        for col in main_columns:
                            if col.lower().endswith("id"):
                                main_id = col
                                break

                    filter_column = None
                    for col in main_columns:
                        if "name" in col.lower():
                            filter_column = col
                            break
                    if not filter_column:
                        filter_column = main_columns[0]  

                    join_tables = [
                        t for t, cols in self.schema.items()
                        if t != main_table and any(col.lower() == main_id.lower() for col in cols)
                    ]

                    print(f"Main table: {main_table}")
                    print(f"Joining tables with {main_id}: {join_tables}")

                    alias_counter = 0
                    join_sql = f"FROM {main_table} m\n"
                    for jt in join_tables:
                        alias = f"t{alias_counter}"
                        join_sql += f"LEFT JOIN {jt} {alias} ON m.{main_id} = {alias}.{main_id}\n"
                        alias_counter += 1

                    sql_query = f"SELECT *\n{join_sql}WHERE m.{filter_column} LIKE '{name_value}';"

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
        print("\nProgram completed.")

if __name__ == "__main__":
    main() 