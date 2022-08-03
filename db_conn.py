import psycopg2
import os
from dotenv import load_dotenv


class DataBase:

    def __init__(self, env_path: str, table: str):
        self.env = self.db_env(env_path)
        self.table = table
        self.connection = psycopg2.connect(dbname=self.env['DB_NAME'],
                                           user=self.env['DB_USER'],
                                           password=self.env['DB_PASS'],
                                           host=self.env['DB_HOST'],
                                           port=self.env['DB_PORT'])
        self.create_table()

    @staticmethod
    def db_env(path):

        dotenv_path = os.path.join(os.path.dirname(__file__), path)
        load_dotenv(dotenv_path)
        return os.environ

    def create_table(self):

        with self.connection:

            cursor = self.connection.cursor()

            create_table_query = f"""CREATE TABLE IF NOT EXISTS {self.table} (
                                             id SERIAL NOT NULL,
                                             time_event timestamp NOT NULL,
                                             license_plate VARCHAR(100) NOT NULL,
                                             image VARCHAR(100) NOT NULL);"""

            cursor.execute(create_table_query)
            self.connection.commit()
            print(f"{self.table} table successfully created in PostgreSQL")

    def update_table(self, update_items: tuple):

        with self.connection:

            cursor = self.connection.cursor()
            insert_query = f"""INSERT INTO {self.table} (id, time_event, license_plate, image) VALUES (%s, %s, %s, %s)"""

            cursor.execute(insert_query, update_items)
            self.connection.commit()
            print(f"{self.table} table successfully update in PostgreSQL")

    def check_last_index(self) -> int:

        with self.connection:

            cursor = self.connection.cursor()
            check_table_idx = f"""SELECT COUNT(*) as count FROM {self.table}"""
            cursor.execute(check_table_idx)
            idx = int(cursor.fetchone()[0])
            return idx
