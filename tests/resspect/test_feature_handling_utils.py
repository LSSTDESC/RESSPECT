import mongomock
import os
import pandas as pd
from pathlib import Path
import tempfile
from unittest.mock import patch

from resspect.feature_handling_utils import *

_TEST_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "tests"
_TEST_MONGO_URI = "mongodb+srv://resspect:0000000V@resspectcluster0.agzec.mongodb.net/"


def test_save_load_from_filesystem():
    with tempfile.TemporaryDirectory() as temp_dir:
        features_path = str(_TEST_DATA_DIR / "test_features.csv")

        df = load_external_features(
            filename=features_path,
            location="filesystem",
            feature_extractor="Malanchev"
        )

        assert len(df) == 51
        assert len(df.columns) == 57

        output_path = str(Path(temp_dir) / "test_df.csv")

        save_features(
            data=df,
            location="filesystem",
            filename=output_path,
            feature_extractor="Malanchev",
        )

        test_df = pd.read_csv(output_path)

        assert len(test_df) == 51
        assert len(df.columns) == 57

def test_save_load_from_mongodb():
    os.environ["MONGO_URI"] = _TEST_MONGO_URI
    test_features_path = str(_TEST_DATA_DIR / "test_features.csv")
    df = pd.read_csv(test_features_path)

    client = mongomock.MongoClient()
    with patch("pymongo.MongoClient.__new__", return_value=client):
        save_features(
            data=df,
            location="mongodb",
            feature_extractor="Malanchev",
        )

        test_df = load_external_features(
            mongo_query={},
            location="mongodb",
            feature_extractor="Malanchev",
        )

        assert len(test_df) == 51
        # column number goes up by one when we save to mongo (mongo object id)
        assert len(test_df.columns) == 58