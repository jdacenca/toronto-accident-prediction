from sklearn.preprocessing import LabelEncoder

# Binary mapping function
def apply_binary_mapping(df, columns, mapping):
    for column in columns:
        df[column] = df[column].map(lambda x: mapping[x.upper()])
    return df

# Target mapping function
def apply_target_mapping(df, column, mapping):
    df[column] = df[column].replace(mapping).astype(int)
    return df

# Label encoding function
def apply_label_encoding(df, columns, encoder=None):
    if encoder is None:
        encoder = LabelEncoder()
    for column in columns:
        df[column] = encoder.fit_transform(df[column])
    return df
