import io
import logging
import os
import pandas as pd
import pymongo

MONGODB_NAME = "resspect_db_test"

MONGO_COLLECTION_NAMES = {
    "Bump": "resspect_bump",
    "Bazin": "resspect_bazin",
    "Malanchev": "resspect_malanchev"
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
        if "MONGO_URI" in os.environ.keys():
            MONGO_URI = os.environ["MONGO_URI"]
        else:
            raise ValueError("Couldn't find $MONGO_URI in envorinment variables.")
        client = pymongo.MongoClient(MONGO_URI)
        db = client[MONGODB_NAME]
        collection = db[MONGO_COLLECTION_NAMES[feature_extractor]]
        data_dicts = [data.loc[i].to_dict() for i in range(len(data))]
        _ = collection.insert_many(data_dicts)
        logging.info(
            "Features have been saved to MongoDB collection: %s",
            MONGO_COLLECTION_NAMES[feature_extractor]
        )
    else:
        raise ValueError("location must either be 'filesystem' or 'location'")

def load_external_features(
        filename: str = None,
        location: str = "filesystem",
        mongo_query: dict = None,
        feature_extractor: str = "Malanchev",
):
    "Load features from a .csv file or download them from a MongoDB instance."
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
    elif location == "mongodb":
        if "MONGO_URI" in os.environ.keys():
            MONGO_URI = os.environ["MONGO_URI"]
        else:
            raise ValueError("Couldn't find $MONGO_URI in envorinment variables.")
        client = pymongo.MongoClient(MONGO_URI)
        db = client[MONGODB_NAME]
        collection = db[MONGO_COLLECTION_NAMES[feature_extractor]]

        cursor = collection.find(mongo_query)
        data_dicts = []
        for element in cursor:
            data_dicts.append(element)
        # Potential TODO: drop the MongoDB `_id` column ?
        data = pd.DataFrame(data_dicts)
    else:
        raise ValueError("location must either be 'filesystem' or 'location'")
    return data
