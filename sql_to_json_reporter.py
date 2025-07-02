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
DB_TYPE = "mysql"  # Changed to mssql for T-SQL
DB_USER = "root"
DB_PASSWORD = "Har@0121"
DB_HOST = "localhost"
DB_PORT = 3306  # Default SQL Server port
DB_NAME = "VMaster7"

# Ollama Configuration
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:4b"  # Using smaller model for faster processing
OLLAMA_TIMEOUT = 120  # 120 seconds timeout

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
        """Convert natural language query to SQL using schema and model instructions"""
        try:
            # Improved extraction of table names
            table_names = []
            query_lower = query.lower()
            # Find the block after 'from', 'of', 'table', or 'in' up to 'where', 'column is', or end of string
            table_block_match = re.search(r'(?:from|of|table|in)\s+([\w\s,]+?)(?:\s+where|\s+column is|$)', query_lower)
            if table_block_match:
                possible_tables = re.split(r'\band\b|,|\s+', table_block_match.group(1))
                table_names = [t.strip() for t in possible_tables if t.strip() in self.schema]
            # Fallback: if not found, use previous logic
            if not table_names:
                table_names = re.findall(r'(?:from|of|table|in)\s+(\w+)', query_lower)
                table_names = [t for t in table_names if t in self.schema]
            
            # Extract requested columns from the query (robust)
            requested_columns = []
            # Look for explicit column list after 'column is', 'columns are', etc.
            columns_match = re.search(r'(?:column is|columns are|where column is|where columns are)\s+([\w,\s]+)', query, re.IGNORECASE)
            if columns_match:
                requested_columns = [col.strip() for col in columns_match.group(1).split(',') if col.strip()]
            # If not found, look for columns after table names (e.g., 'table tbl1, tbl2 col1, col2')
            elif len(table_names) > 0:
                # Try to find columns after the last table name
                after_tables = query.lower().split(table_names[-1].lower(), 1)[-1]
                possible_cols = re.findall(r'([a-zA-Z_][\w\s,]*)', after_tables)
                if possible_cols:
                    # Take the first match and split by comma
                    first_cols = possible_cols[0]
                    if any(c.isalpha() for c in first_cols):
                        requested_columns = [col.strip() for col in first_cols.split(',') if col.strip()]
            # If still not found, use * (all columns)
            if not requested_columns:
                requested_columns = ['*']
            print(f"DEBUG: Extracted tables: {table_names}")
            print(f"DEBUG: Requested columns: {requested_columns}")
            
            if len(table_names) == 2:
                # Prepare schema info for both tables
                schema_info = json.dumps({t: self.schema[t] for t in table_names}, indent=2)
                columns_str = ', '.join(requested_columns) if requested_columns and requested_columns != ['*'] else '*'
                # Detect if a join is implied by the query (look for 'based on', 'join', or common key like CustID)
                join_key = None
                for key in ['custid', 'customerid', 'id']:
                    if all(any(key in col.lower() for col in self.schema[t]) for t in table_names):
                        join_key = key
                        break
                join_detected = False
                if re.search(r'based on|join|matching|with same|using', query.lower()) or join_key:
                    join_detected = True
                if join_detected and join_key:
                    prompt = f"""Generate a SQL query that joins the tables: {', '.join(table_names)}.
The columns to include are: {columns_str}.

User request: {query}

Available tables and columns:
{schema_info}

Instructions:
- Join the tables on the common key: {join_key}.
- Select the requested columns from each table, in the same order as listed.
- If a column does not exist in a table, use NULL AS column_name for that table.
- Do not include semicolons in the query.
"""
                else:
                    prompt = f"""Generate a SQL query that merges data from both tables: {', '.join(table_names)}.
The columns to include are: {columns_str}.

User request: {query}

Available tables and columns:
{schema_info}

Instructions:
- For each SELECT, use only the requested columns in the same order.
- If a column does not exist in a table, use NULL AS column_name for that table.
- Merge the results using UNION ALL.
- Do not include semicolons in the query.
- Use simple SELECT statements.

Example:
SELECT col1, col2 FROM table1
UNION ALL
SELECT col1, NULL AS col2 FROM table2
"""
                try:
                    response = client.generate(
                        model=OLLAMA_MODEL,
                        prompt=prompt,
                        stream=False,
                        system="You are a SQL query generator. Generate ONLY the SQL query."
                    )
                    sql_query = response['response'].strip()
                    sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
                    sql_query = sql_query.rstrip(';')
                    sql_query = re.sub(r'\s+LIMIT\s+\d+', '', sql_query, flags=re.IGNORECASE)
                    return sql_query
                except Exception as e:
                    print(f"Model error: {str(e)}")
                    # Fallback to simple query
                    return f"SELECT * FROM {table_names[0]} UNION ALL SELECT * FROM {table_names[1]}"
            
            # Fallback to original logic for single table
            table_name = None
            table_patterns = [
                (r'from\s+(\w+)', 'from'),
                (r'of\s+(\w+)', 'of'),
                (r'table\s+(\w+)', 'table'),
                (r'in\s+(\w+)', 'in')
            ]
            for pattern, _ in table_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    table_name = match.group(1).strip()
                    break
            
            # Check if table exists in schema
            if table_name and table_name in self.schema:
                # Define common condition patterns
                condition_patterns = {
                    'on hold': ('OnHold = true', 'OnHold = false'),
                    'active': ('Inactive = false', 'Inactive = true'),
                    'inactive': ('Inactive = true', 'Inactive = false'),
                    'vendor': ('IsVendor = true', 'IsVendor = false'),
                    'non vendor': ('IsVendor = false', 'IsVendor = true'),
                    'credit approved': ('CreditAuth = true', 'CreditAuth = false'),
                    'not credit approved': ('CreditAuth = false', 'CreditAuth = true'),
                    'locked': ('LockAccount = true', 'LockAccount = false'),
                    'not locked': ('LockAccount = false', 'LockAccount = true')
                }
                
                # Check for condition patterns
                for pattern, (true_condition, false_condition) in condition_patterns.items():
                    if pattern in query_lower:
                        # Check for negation
                        if any(neg in query_lower for neg in ['not', 'no', 'non', 'without']):
                            return f"SELECT * FROM {table_name} WHERE {false_condition}"
                        else:
                            return f"SELECT * FROM {table_name} WHERE {true_condition}"
                
                # Handle date-based queries
                date_patterns = {
                    'today': "CustDateEntered = CURDATE()",
                    'yesterday': "CustDateEntered = DATE_SUB(CURDATE(), INTERVAL 1 DAY)",
                    'this week': "CustDateEntered >= DATE_SUB(CURDATE(), INTERVAL WEEKDAY(CURDATE()) DAY)",
                    'this month': "MONTH(CustDateEntered) = MONTH(CURDATE()) AND YEAR(CustDateEntered) = YEAR(CURDATE())",
                    'this year': "YEAR(CustDateEntered) = YEAR(CURDATE())"
                }
                
                for pattern, condition in date_patterns.items():
                    if pattern in query_lower:
                        return f"SELECT * FROM {table_name} WHERE {condition}"
                
                # Handle numeric comparisons
                numeric_patterns = {
                    r'credit limit (?:greater than|more than|above) (\d+)': lambda x: f"CreditLimit > {x}",
                    r'credit limit (?:less than|below|under) (\d+)': lambda x: f"CreditLimit < {x}",
                    r'credit limit (?:equal to|exactly) (\d+)': lambda x: f"CreditLimit = {x}",
                    r'discount (?:greater than|more than|above) (\d+)': lambda x: f"CustDiscount > {x}",
                    r'discount (?:less than|below|under) (\d+)': lambda x: f"CustDiscount < {x}",
                    r'discount (?:equal to|exactly) (\d+)': lambda x: f"CustDiscount = {x}"
                }
                
                for pattern, condition_func in numeric_patterns.items():
                    match = re.search(pattern, query_lower)
                    if match:
                        value = match.group(1)
                        return f"SELECT * FROM {table_name} WHERE {condition_func(value)}"
                
                # Handle text-based searches
                text_patterns = {
                    r'name (?:like|contains|has) "([^"]+)"': lambda x: f"CustName LIKE '%{x}%'",
                    r'email (?:like|contains|has|is) "([^"]+)"': lambda x: f"Email = '{x}'",
                    r'type (?:is|equals) "([^"]+)"': lambda x: f"CustType = '{x}'",
                    r'group (?:is|equals) "([^"]+)"': lambda x: f"GroupVal = '{x}'"
                }
                
                # Handle email directly in the query
                email_match = re.search(r'(\S+@\S+\.\S+)', query)
                if email_match:
                    email = email_match.group(1)
                    return f"SELECT * FROM {table_name} WHERE Email = '{email}'"
                
                for pattern, condition_func in text_patterns.items():
                    match = re.search(pattern, query_lower)
                    if match:
                        value = match.group(1)
                        return f"SELECT * FROM {table_name} WHERE {condition_func(value)}"
                
                # If no specific patterns match, use the model for complex queries
                schema_info = json.dumps({table_name: self.schema[table_name]}, indent=2)
                
                prompt = f"""Generate a SQL query for: {query}

Available tables and columns:
{schema_info}

Instructions:
1. Use simple SELECT statements
2. Place WHERE clause before LIMIT
3. Use proper table names
4. For date filtering in MySQL, use appropriate date functions
5. Keep the query simple and direct
6. Do not include semicolons in the query

Example: SELECT * FROM table_name WHERE MONTH(date_column) = 6 LIMIT 1000"""

                try:
                    response = client.generate(
                        model=OLLAMA_MODEL,
                        prompt=prompt,
                        stream=False,
                        system="You are a SQL query generator. Generate ONLY the SQL query."
                    )
                    
                    sql_query = response['response'].strip()
                    sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
                    sql_query = sql_query.rstrip(';')
                    
                    # Remove LIMIT clause if present
                    sql_query = re.sub(r'\s+LIMIT\s+\d+', '', sql_query, flags=re.IGNORECASE)
                    
                    return sql_query
                    
                except Exception as e:
                    print(f"Model error: {str(e)}")
                    # Fallback to simple query
                    return f"SELECT * FROM {table_name}"
                    
        except Exception as e:
            print(f"Error generating SQL: {str(e)}")
            raise

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
        """Process natural language query and return results, with auto-join for addresses and custcontact if mentioned."""
        sql_query = self.natural_language_to_sql(natural_language_query)
        print(f"\nGenerated SQL: {sql_query}")

        # --- Auto-join logic for addresses and custcontact ---
        query_lower = natural_language_query.lower()
        needs_addresses = 'addresses' in query_lower
        needs_custcontact = 'custcontact' in query_lower
        if (needs_addresses or needs_custcontact):
            # Build SELECT clause with all requested customer columns
            requested_columns = [
                'CustID', 'CustName', 'Email', 'CustBranchID', 'CustAccountRep', 'CustType', 'CustSource',
                'OrderFrequency', 'ProjectedRevenue', 'CreditLimit', 'CustTerms', 'CustDiscount', 'CustNotes', 'UrgentNotes'
            ]
            select_cols = [f"c.{col}" for col in requested_columns]
            join_clauses = []
            if needs_addresses and 'addresses' in self.schema:
                address_cols = [col for col in self.schema['addresses'] if col.lower() != 'custid']
                select_cols += [f"a.{col}" for col in address_cols]
                join_clauses.append("LEFT JOIN addresses a ON c.CustID = a.CustID")
            if needs_custcontact and 'custcontact' in self.schema:
                contact_cols = [col for col in self.schema['custcontact'] if col.lower() != 'custid']
                select_cols += [f"cc.{col}" for col in contact_cols]
                join_clauses.append("LEFT JOIN custcontact cc ON c.CustID = cc.CustID")
            select_clause = ',\n  '.join(select_cols)
            join_clause = '\n'.join(join_clauses)
            sql_query = f"SELECT\n  {select_clause}\nFROM customers c\n{join_clause}"
            print(f"\nAuto-joined SQL: {sql_query}")

        results = self.execute_query(sql_query)
        
        if isinstance(results, dict) and "error" in results:
            print(f"\nError: {results['error']}")
            return None
        
        if not results:
            # Check if this was a user detail query that returned no results
            if 'user_id' in natural_language_query.lower() or 'user id' in natural_language_query.lower():
                print("\nNo user found with the specified ID.")
            else:
                print("\nNo data found matching your query.")
            return None
        
        # Format results in flat array structure
        formatted_results = []
        for row in results:
            # Always add table_name if it's not present
            if 'table_name' not in row:
                table_name = self.get_table_name(sql_query)
                new_row = {'table_name': table_name}
                new_row.update(row)  # Add all other fields after table_name
                formatted_results.append(new_row)
            else:
                formatted_results.append(row)
        
        # Save results to file
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