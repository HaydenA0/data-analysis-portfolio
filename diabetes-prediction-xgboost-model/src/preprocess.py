import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

raw = pd.read_csv("Project/data/raw.csv")


def clean(data):
    return data[data["gender"] != "Other"]


def numerization(data):
    object_columns = data.select_dtypes("object").columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(data[object_columns])
    encoded_array = encoder.transform(data[object_columns])
    encoded_columns = encoder.get_feature_names_out(object_columns)

    encoded_df = pd.DataFrame(encoded_array, columns=encoded_columns, index=data.index)
    non_cat_data = data.drop(columns=object_columns)
    final_df = pd.concat([non_cat_data, encoded_df], axis=1)

    return final_df


def split_data(data, val_size=0.25):
    columns = list((data).columns)
    columns.remove("diabetes")
    X = data[columns]
    y = data.diabetes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=val_size * 2, random_state=42
    )

    return X_train, X_test, y_train, y_test
