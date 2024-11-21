import io
import logging
import pandas as pd
import tarfile

def save_features(
        data: pd.DataFrame,
        location: str = "filesystem",
        filename: str = None,
):
    """Save features from a pandas Dataframe."""
    if location == "filesystem":
        if filename is not None:
            data.to_csv(filename)
            logging.info("Features have been saved to: %s", filename)
        else:
            raise ValueError("filename must be provided if saving to the filesystem.")
    else:
        raise NotImplementedError("Alternative storage method implementation tbd.")

def load_external_features(
        filename: str = None,
        location: str = "filesystem",
):
    "Load features from a .csv file."
    data = None
    if location == "filesystem":
        if filename is not None:
            if '.tar.gz' in filename:
                tar = tarfile.open(filename, 'r:gz')
                fname = tar.getmembers()[0]
                content = tar.extractfile(fname).read()
                data = pd.read_csv(io.BytesIO(content))
                tar.close()
            else:
                data = pd.read_csv(filename, index_col=False)
                print(data.keys()[0])
                if "Unnamed" not in data.keys()[0] and " " in data.keys()[0]:
                    data = pd.read_csv(filename, sep=' ', index_col=False)
        else:
            raise ValueError("filename must be provided if saving to the filesystem.")
    else:
        raise NotImplementedError("Alternative storage method implementation tbd.")
    return data