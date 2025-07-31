import os
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from dotenv import load_dotenv

load_dotenv()

class MongoDBClient:
    """
    MongoDB 와 연결을 한번만 하도록 하는 싱글톤 구조
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MongoDBClient, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        # The __init__ is called every time, but the connection is only made once.
        if not hasattr(self, 'client'):
            mongo_uri = os.getenv("MONGO_URI")
            if not mongo_uri:
                raise ValueError("MONGO_URI environment variable not set.")
            
            self.client: MongoClient = MongoClient(mongo_uri)
            self.db: Database = self.client.get_database("AirBot")
            print("MongoDB connection established.")

    def get_collection(self, collection_name: str) -> Collection:
        return self.db.get_collection(collection_name)


mongo_client = MongoDBClient()

airport_congestion_t1_collection = mongo_client.get_collection("AirportCongestionNow_T1")
airport_congestion_t2_collection = mongo_client.get_collection("AirportCongestionNow_T2")
aiport_congestion_predict = mongo_client.get_collection("AirportCongestionPredict")

