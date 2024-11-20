import logging
import pandas as pd

def save_features(
        data: pd.DataFrame,
        location: str = "filesystem",
        filename: str = None,
):
    if location == "filesystem":
        if filename is not None:
            data.to_csv(filename)
            logging.info("Features have been saved to: %s", filename)
        else:
            raise ValueError("filename must be provided if saving to the filesystem.")
    else:
        raise NotImplementedError("Alternative storage method implementation tbd.")