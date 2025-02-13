import urllib.request
import zipfile
import os
from pathlib import Path 
import pandas as pd

def download_and_unzip_spam_data(
    url: str , zip_path: str, extracted_path: str, data_file_path: str):
    """downloads spam data"""
    
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download")
        return
        
    #Download the file and write into zip_path
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    #Unzip the file into direcgtory of extracted path
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    # add .tsv to original_file_path name
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path) 
    print(f"File downloaded and saved as {data_file_path}")

def undersample_dataset(
    df: pd.DataFrame, smaller_freq_label: str, higher_freq_label: str):

    smaller_freq_label_subset = df[df["Label"] == smaller_freq_label]
    n_smaller_freq_label = smaller_freq_label_subset.shape[0]
    higher_freq_label_subset = df[df["Label"] == higher_freq_label].sample(
        n_smaller_freq_label, random_state=42
    )
    balanced_df = pd.concat([
        higher_freq_label_subset, smaller_freq_label_subset
    ])

    return balanced_df

def random_split(df: pd.DataFrame, train_frac: float, val_frac: float):
    #Shuffle entire dataset
    df = df.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)


    train_end_idx = int(len(df) * train_frac)
    val_end_idx = train_end_idx + int(len(df) * val_frac)
    train_df = df[:train_end_idx]
    validation_df = df[train_end_idx:val_end_idx]
    test_df = df[val_end_idx:]

    return train_df, validation_df, test_df

def create_ftuning_datasets():
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extracted_path = "sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

    download_and_unzip_spam_data(
        url, zip_path, extracted_path, data_file_path
    )

    df = pd.read_csv(
        data_file_path, sep="\t", header = None, names=["Label", "Text"]
    )
    
    balanced_df = undersample_dataset(
        df=df,
        smaller_freq_label="spam",
        higher_freq_label="ham"
    )

    train_df, validation_df, test_df = random_split(
        balanced_df, train_frac=0.7, val_frac=0.1
    )
    
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)
    print("saved train, validation and text data")
            
if __name__ == "__main__":
    create_ftuning_datasets()
