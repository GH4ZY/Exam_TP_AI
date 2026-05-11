import os
import sqlite3
import pandas as pd

#Connects to (or creates) mmm.db.
csv_directory = 'data' 
database_name = 'mmm.db'        
conn = sqlite3.connect(database_name)


#Loops over each CSV file in the /data directory.

for filename in os.listdir(csv_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(csv_directory, filename)
        table_name = os.path.splitext(filename)[0] 
        #Converts each CSV to a Pandas dataframe, then inserts it into SQLite under a table named after the file.
        df = pd.read_csv(file_path)

        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Imported '{filename}' into table '{table_name}'.")

conn.close()
