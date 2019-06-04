import pandas as pd
import time
import re

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

__all__ = ['DataFilter']