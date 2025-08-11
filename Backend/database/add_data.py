

from Backend.database.migrate import load_data_from_csvs

def load():
    print("adding data to the database...")
    load_data_from_csvs()
    print("Data added successfully.")

if __name__ == "__main__":
    load()