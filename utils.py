from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder



def load_data(filename_arff):
    dataset = arff.loadarff(filename_arff)
    data = dataset[0]
    df = pd.DataFrame(data)
    input_data_df = df.iloc[:, 0:3052]

    # creating instance of one-hot-encoder
    encoder = OneHotEncoder(handle_unknown='ignore')

    # perform one-hot encoding on 'team' column
    labels_df = pd.DataFrame(encoder.fit_transform(df[['target']]).toarray())
    labels_df.columns = ['Artifact', 'ExtraHLS', 'ExtraStole', 'Murmur', 'Normal']
    return input_data_df, labels_df


def get_multiscale_lengths(dataframes):
    lengths = []

    for df in dataframes:
        # Get the size of the DataFrame
        num_rows, num_cols = df.shape

        lengths.append(num_cols)
    return lengths











