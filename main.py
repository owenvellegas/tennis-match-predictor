from src.data_processing import load_data, preprocess_data, process_data
from src.model.train_model import train_model

def main():
    files = [
        "atp_matches_2013.csv",
        "atp_matches_2014.csv",
        "atp_matches_2015.csv",
        "atp_matches_2016.csv",
        "atp_matches_2017.csv",
        "atp_matches_2018.csv",
        "atp_matches_2019.csv",
        "atp_matches_2020.csv",
        "atp_matches_2021.csv",
        "atp_matches_2022.csv",
        "atp_matches_2023.csv",
        "atp_matches_2024.csv"
    ]

    raw_df = load_data(files, data_dir="data")

    preprocessed_df = preprocess_data(raw_df)

    processed_df = process_data(preprocessed_df)

    train_model(processed_df)


if __name__ == "__main__":
    main()