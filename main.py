from src.data_loader import load_data, preprocess_data

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

    processed_df = preprocess_data(raw_df)

    print(processed_df.head())


if __name__ == "__main__":
    main()