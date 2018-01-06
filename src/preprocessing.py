import pandas as pd

def get_tree_data():
    """
    Creates a dataframe with the street tree data

    OUTPUT:
    - df (dataframe): contains information in the street tree dataset
    """
    df = pd.read_csv('data/Street_Tree_List.csv')

    # Ignore trees without geocodes
    df = df[~pd.isnull(df['Location'])]

    # Convert planting dates to datetime
    df['PlantDate'] = pd.to_datetime(df['PlantDate'])

    # Exclude trees planted in or after April 2011 (imagery vintage)
    df = df[~((df['PlantDate'].dt.year == 2011) & (df['PlantDate'].dt.month > 3))]
    df = df[~(df['PlantDate'].dt.year > 2011)]

    return df

def get_imagery_metadata():
    """
    Creates a dataframe with the street tree data

    OUTPUT:
    - df (dataframe): contains information in the street tree dataset
    """
    df = pd.read_csv('data/HIGH_RES_ORTHO_227914.txt',
                      usecols = ['Image Name',
                                 'NW Corner Lat dec', 'NW Corner Long dec',
                                 'SE Corner Lat dec', 'SE Corner Long dec',])

    return df
