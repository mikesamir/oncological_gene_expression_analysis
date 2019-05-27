# Contains classes: DataFilter, DataPrep, FeatureSelection, Evaluation, FilterResults

class DataFilter:
    """
    Filter dataset to retrieve cancer-specific chunks of data.
    
    Main function -> split(sites, types=["Primary Tumor", "Normal Tissue"], genders=["Female", "Male"], categories=None)
    - Input: Site, tissue type, gender, detailed category.
    - Return: Filtered dataframe. Also saves to csv.
    
    Included functions:
    - init: Define class variables.
    - get_columns: Select for desired criteria.
    - get_output_path: Create filepath for dataframe created by split.
    """
    def __init__(self, data_filepath, phenotype_filepath, upcg=None, data_output='Output'):
        self.data_output = data_output
        self.data_filepath = data_filepath
        self.phenotype_filepath = phenotype_filepath
        self.upcg = upcg
        import pandas as pd
        self.df_phenotype = pd.read_csv(phenotype_filepath, delimiter='\t')

        
    def get_columns(self, sites, types=["Primary Tumor", "Normal Tissue"], genders=["Female", "Male"], categories=None):
        """
        - Select for cancer-specific criteria
        - Returns dataframe
        """
        cond = self.df_phenotype['_primary_site'].isin(sites)
        cond &= self.df_phenotype['_gender'].isin(genders)
        cond &= self.df_phenotype['_sample_type'].isin(types)
        
        if categories != None:
            cond &= self.df_phenotype['detailed_category'].isin(categories)

        return ['sample'] + self.df_phenotype[cond]['sample'].tolist()
    
    
    def get_output_path(self, sites, categories):
        """
        - Create filepath for dataframe created by split.
        - Uses self.data_ouput which was defined when instantiating the class object.
        """
        items = sites if categories == None else categories
        name = '_'.join([re.sub('\s', '', s) for s in items]) 
        
        return '{}/Chunk_{}.csv'.format(self.data_output, name)

    
    def split(self, sites, **kwargs):
        """
        - Combine functions: get_columns, get_output_path
        - Input: Site, tissue type, gender, detailed category.
        - Return: Filtered dataframe. Also saves to csv.
        """
        start = time.time()

        # Filter columns to load a smaller data subset.
        filtered_cols = self.get_columns(sites, **kwargs)

        print('Processing data with {} samples.'.format(len(filtered_cols)))
        
        # Load the original big fat data file with specific columns for a final chunk.
        df = pd.read_csv(self.data_filepath, delimiter='\t', usecols=filtered_cols)
        
        # Filter all ensembls that are not representing protein-coding genes.
        if self.upcg != None:
            df = df[df['sample'].str.replace(r'\.\d+', '').isin(self.upcg)]

        # Transpose the matrix to get ensembls as headers.
        df = df.set_index('sample').transpose()
        df.insert(loc=0, column='label', value=df.index.astype(str).str.contains('TCGA-').astype(int))
        
        # Store as csv.
        output_path = self.get_output_path(sites, kwargs.get('categories'))
        df.to_csv(output_path)
        print('Finished in {:.1f} min\n'.format((time.time() - start) / 60))
        
        return df

    
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
    
    
    # Read Data
    def read_data(self, path, nrows, usecols):
        """
        - Reads .tsv file
        - Drops unnecessary columns
        - Removes Ensembl version notation
        """
        data = pd.read_csv(path, nrows=nrows, usecols=usecols)
        data.index = data.iloc[:,0]
        data.drop(columns = "Unnamed: 0", inplace = True)
        data.columns = [(re.sub('\.\d+', '', gene)) for gene in data.columns]
        
        return data
    

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
    

    def bulbasaur(self, path, threshold = 2, nrows = None, usecols = None):
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
    
    