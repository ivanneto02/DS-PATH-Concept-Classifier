from sklearn.model_selection import train_test_split
from config import DATA_PATH, DATA_FILE, NROWS
import keras
import pandas as pd
from sklearn.utils import shuffle

def main():
    pd.options.display.max_rows = 100

    # Read in data
    print("> Reading dataset...")
    df = pd.read_csv(DATA_PATH + "/" + DATA_FILE, nrows=NROWS)

    # Shuffle data
    print("> Shuffling dataset...")
    df = shuffle(df)

    # Splitting
    print("> Performing train/test split...")
    train, test = train_test_split(df, test_size=0.3) # 70% train
    test, val = train_test_split(test, test_size=0.3) # 20% test, 10% val
    
    column_index = 7

    x_train, y_train = train.iloc[:, column_index:], train["concept_type"]
    x_test, y_test = test.iloc[:, column_index:], test["concept_type"]
    x_val, y_val = val.iloc[:, column_index:], val["concept_type"]

    # print(x_train)

    print(y_train.head(100))

if __name__ == "__main__":
    main()