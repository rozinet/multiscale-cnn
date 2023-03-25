from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_data(filename_arff):
    dataset = arff.loadarff(filename_arff)
    data = dataset[0]
    df = pd.DataFrame(data)
    # creating instance of one-hot-encoder
    encoder = OneHotEncoder(handle_unknown='ignore')

    # perform one-hot encoding on 'team' column
    labels_df = pd.DataFrame(encoder.fit_transform(df[['target']]).toarray())
    labels_df.columns = ['Artifact', 'ExtraHLS', 'ExtraStole', 'Murmur', 'Normal']
    input_data_df = df.iloc[:, 0:3052]

    # filter only for 3 classes
    # input_data_df = df.iloc[40:, 0:3052]
    # labels_df = labels_df[['ExtraStole', 'Murmur', 'Normal']].iloc[40:, :]
    return input_data_df, labels_df

def get_multiscale_lengths(dataframes):
    lengths = []

    for df in dataframes:
        # Get the size of the DataFrame
        num_rows, num_cols = df.shape

        lengths.append(num_cols)
    return lengths











