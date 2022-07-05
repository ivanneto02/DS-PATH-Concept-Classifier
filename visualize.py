
from config import DATA_PATH, DATA_FILE, NROWS
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib as mpl

def main():

    plt.style.use('seaborn')

    # pd.options.display.max_rows = 100
    # pd.options.display.max_columns = 10

    # Read in data
    print("> Reading DataFrame...")
    df = pd.read_csv(DATA_PATH + "/" + DATA_FILE, nrows=None)

    print("> Shuffling...")
    df = shuffle(df)

    print("> Fairly splitting...")
    df_dis = df[df["concept_type"] == "disease"]
    df_dru = df[df["concept_type"] == "drug"]
    n = min(len(df_dru), len(df_dis))
    df_dis = df_dis[:n//2]
    df_dru = df_dru[:n//2]
    df = shuffle(pd.concat([df_dis, df_dru]))

    print("> Plotting...")
    df["concept_type"].value_counts().plot(kind="bar")
    plt.show()

if __name__ == "__main__":
    main()