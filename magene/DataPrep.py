import pandas as pd
import re

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

class DataPrep:
    """
    Prepare data for feature selection algorithm.
    
    Main function -> bulbasaur(path, threshold, nrows = None, usecols = None)
    - input: Directory path of gene expression data.
    - output: X_train, y_train, x_test, y_test.

    Included functions:
    - read_data.
    - X_and_y.
    - split: Train and test split.
    - smote_up: Upsampling to get balanced dataset.
    """
    def __init__(self, seed):
        self.seed = seed
    
    def read_data(self, path, nrows, usecols):
        """
        - Reads .tsv file
        - Drops unnecessary columns
        - Removes Ensembl version notation
        """
        df = pd.read_csv(path, nrows=nrows, usecols=usecols)
        df.index = df.iloc[:,0]
        
        if 'Unnamed: 0' in df.columns:
            df.drop(columns = "Unnamed: 0", inplace = True)
        
        df.columns = [(re.sub('\.\d+', '', gene)) for gene in df.columns]
        
        return df
    
    # Filter with Standard Deviation Threshold
    def X_and_y(self, data, threshold):
        """
        - Defines features (expression values) and target (sample type)
        - Filters expression values with a standard deviation threshold
        """
        X = data.drop(columns = 'label')
        X_sd = X.loc[:, X.std() > threshold]
        y = data[["label"]]
        
        return X_sd, y
    

    def split(self, X, y, test_size):
        """
        - Split data into train and test
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.seed)
        
        return X_train, X_test, y_train, y_test
    
    def smote_up(self, X_train, y_train):
        """
        - Upsample data where needed using SMOTE
        """
        sm = SMOTE(random_state=self.seed)
        X_train_smote, y_train_smote = sm.fit_sample(X_train, y_train)
        column_names = X_train.columns

        # Make dataframe again
        X_train_smote = pd.DataFrame(X_train_smote, columns=column_names)
        y_train_smote = pd.DataFrame(y_train_smote, columns=['label'])

        return X_train_smote, y_train_smote
    
    def bulbasaur(self, path, threshold=2, nrows=None, usecols=None):
        """
        - Combine functions: read_data, X_and_y, split, smote_up
        - input: directory path of gene expression data
        - output: X_train, y_train, x_test, y_test
        """
        data = self.read_data(path, nrows, usecols)
        X, y = self.X_and_y(data, threshold)
        X_train, X_test, y_train, y_test = self.split(X, y, 0.3)
        X_train_smote, y_train_smote = self.smote_up(X_train, y_train)

        return X_train_smote, y_train_smote, X_test, y_test

__all__ = ['DataPrep']