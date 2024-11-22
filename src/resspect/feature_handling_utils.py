import io
import logging
import os
import pandas as pd
from pymongo.mongo_client import MongoClient

MONGODB_NAME = "resspect_db_test"

MONGO_COLLECTION_NAMES = {
    "Bump": "resspect_bump"
}
import tarfile

def save_features(
        data: pd.DataFrame,
        location: str = "filesystem",
        filename: str = None,
        feature_extractor: str = "Malanchev",
):
    """Save features from a pandas Dataframe."""
    if location == "filesystem":
        if filename is not None:
            data.to_csv(filename)
            logging.info("Features have been saved to: %s", filename)
        else:
            raise ValueError("filename must be provided if saving to the filesystem.")
    elif location == "mongodb":
        with open("~/homework/mongodb_test.pass") as f:
            MONGO_URI = f.readline().strip("\n")
        client = MongoClient(MONGO_URI)
        db = client[MONGODB_NAME]
        collection = db[MONGO_COLLECTION_NAMES[feature_extractor]]
        data_dicts = [data.loc[i].to_dict() for i in range(len(data))]
        collection.insert_many(data_dicts)
        logging.info(
            "Features have been saved to MongoDB collection: %s",
            MONGO_COLLECTION_NAMES[feature_extractor]
        )
    else:
        raise ValueError("location must either be 'filesystem' or 'location'")

def load_external_features(
        filename: str = None,
        location: str = "filesystem",
):
    "Load features from a .csv file."
    data = None
    if location == "filesystem":
        if filename is not None:
            if '.tar.gz' in filename:
                with tarfile.open(filename, 'r:gz') as tar:
                    fname = tar.getmembers()[0]
                    content = tar.extractfile(fname).read()
                    data = pd.read_csv(io.BytesIO(content))
            else:
                data = pd.read_csv(filename, index_col=False)
                print(data.keys()[0])
                if "Unnamed" not in data.keys()[0] and " " in data.keys()[0]:
                    data = pd.read_csv(filename, sep=' ', index_col=False)
        else:
            raise ValueError("filename must be provided if reading from the filesystem.")
    else:
        raise NotImplementedError("Alternative storage method implementation tbd.")
    return data
