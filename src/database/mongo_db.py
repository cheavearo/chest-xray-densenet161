# Setup
import pymongo
from bson import ObjectId
import gridfs
import os
from datetime import  datetime
import io
from PIL import Image # pillow in 'requirements.txt'
from dotenv import load_dotenv
load_dotenv()

import certifi
'''
Note: certifi is a Python package that make
root of certifikit for a secure connection.
'''
ca=certifi.where()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class MongoDBNetwork():
    def __init__(self, database_name="VEAROAI", mongo_db_url=MONGO_DB_URL):     
        self.mongo_client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
        self.db = self.mongo_client[database_name]
        self.fs = gridfs.GridFS(self.db)

        # Initiate creating index
        self.create_patient_index()

    def save_inferenced_image(self, output_img_path, patient_id, prediction, conf_score):
        """A function to save inferenced image to Mongo DB."""
        with open(output_img_path, "rb") as image_file:
            file_id = self.fs.put(
                image_file,
                filename=os.path.basename(output_img_path),
                contentType ="image/png",
                metadata = {
                    "patient_id" : patient_id,
                    "prediction" : prediction,
                    "confidence" : float(conf_score),
                    "model_version": "best_full_finetune_4_classes_model",
                    "timestamp" : datetime.utcnow()
                }
            )
        print(f"Image uploaded successfully. File ID : {file_id}")
        return file_id
    
    def create_patient_index(self):
        """ Creates index on metadata.patient_id"""
        self.db["fs.files"].create_index(
            [("metadata.patient_id", pymongo.ASCENDING)],
            name = "patient_id_index"
        )
        print("Patient ID index ensured.")
        
    def list_files(self):
        files = self.fs.find()
        print("List File in GridFS :")
        for file in files:
            pid = file.metadata.get("patient_id") if file.metadata else None
            print(f"Filename : {file.filename} | Patient : {pid} | ID : {file._id}")

    def download_images(self, file_id, output_path):
        grid_out = self.fs.get(ObjectId(file_id))
        with open(output_path, "wb") as f:
            f.write(grid_out.read())

        print("Image downloaed successfully.")
    def find_patient_id(self, patient_id):
        """
        Find a patient by his/her ID.
        Augs:
            patient_id (str) : A recorded patient ID.
        Return:
            metadata list (dict) : A metadata.
        """
        files = self.db["fs.files"].find({"metadata.patient_id": patient_id}).sort("uplaodDate", -1)
        scans = []

        for file in files:
            scans.append({
                "patient_id" : str(file["_id"]),
                "filename" : file.get("filename"),
                "prediction" : file["metadata"].get("prediction"),
                "confidence" : float(file["metadata"].get("confidence")),
                "timestamp" : file["metadata"].get("timestamp")
            })
        return scans
    
    def load_image_from_database(self, file_id):
        """Load images from Mongo DB."""
        grid_out= self.fs.get(ObjectId(file_id))
        # Read image bytes from database
        image_bytes = grid_out.read()
        # Convert image_bytes into bytes IO and using PIL to open those image.
        return Image.open(io.BytesIO(image_bytes))

            

    
