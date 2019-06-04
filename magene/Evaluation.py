import pandas as pd
import re
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import magene

class Evaluation:
    """
    Stores results (Counts of genes selected, importances, overlaps with cosmic genes) in a dataframe. 
    Cosmic: https://cancer.sanger.ac.uk/cosmic - you need a student e-mail to register and download data.
    
    Main function -> iterate_through_cancers(path_list, path_intogen_list, nrows, usecols, threshold = 2.5)
    - Input: path_list (filepath to expression data chunks), path_intogen_list (filepath to cosmic data), threshold for standard deviation filter.
    - Output: Stores results as csv files.
    
    Included functions:
    - Add Cosmic to Dict: Add a dictionary with Cosmic cancer-related genes.
    - Results: Store results as csv file
    - Normalize Importances: Normalize of importances and add column with Total Importance
    - Final Results: Everything together in one df :)
    - Iterate Through Cancers: Iterate throuhg all cancer data
    """    
    def __init__(self, seed):
        self.seed = seed

    def add_cosmic_to_dict(self, path, dict_list):
        """
        - Add a dictionary with Cosmic cancer-related genes and corresponding Mutation Count.
        """
        #path = "Data/Intogen_Data/Lung_Adenocarcinoma_LUAD_TCGA.tsv"
        census = pd.read_csv("Data/Reference_Data/Census_allWed May 15 09_46_55 2019.csv")
        census["GENE"] = census["Synonyms"].str.extract(pat='(ENSG...........)')
        census = census[census["GENE"].notnull()]
        census.fillna("None", inplace = True)
        importances = census["Role in Cancer"].tolist()
        cosmic_genes = {"Cosmic":[census["GENE"].to_list(), importances]}
        dict_list = { **dict_list, **cosmic_genes }
        
        return dict_list
    
    def results(self, dict_list):
        """
        - Store results in a csv.
        - Results nclude: Count for each method, feature importance, intogen counts.
        """
        row_names = []
        column_names = []

        # Create dataframe with all viable genes from all method results
        for method, selected_features in dict_list.items():
            
            for feature in selected_features[0]:
                row_names.append(feature)

            row_names = list(set(row_names))
            column_names.append(method)

        results = pd.DataFrame(columns = column_names, index = row_names)
        results.fillna(0, inplace = True)
        
        # Add a one where the method selected the corresponding feature
        for method, selected_features in dict_list.items():
            for feature in selected_features[0]:
                results.at[feature, method] = 1

        # Create Column with total count
        results['Total Count'] = results[list(results.columns)].sum(axis=1)
        results.sort_values(by = "Total Count", ascending = False, inplace = True)
        
        # Add Importance Columns
        for method, selected_features in dict_list.items():
            additional = pd.DataFrame({"Importances: " + method:selected_features[1]}, index = selected_features[0])
            results = results.join(additional, how="outer")
            """if "key_0" in results.columns:
                results.drop(columns = "key_0", inplace = True)"""
        
        # Clean dataframe
        results.fillna(0, inplace = True)
        results = results.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
        results.rename(index=str, columns={"Importances: Cosmic": "Role in Cancer"}, inplace = True)
        
        return results
    
    def normalize_importance(self, result, threshold = 0):
        """
        This function returns the results table with normalized importances and an extra column - Total importance
        It is then sorted by total importance
        """
        #Normalize Importances
        imp = ['Importances: Gradient Boost Classifier',
               'Importances: Recursive Feature Elimination',
               'Importances: Elastic Net', 
               'Importances: Boruta Tree',
               'Importances: Lasso CV']
        
        # Create scalers
        scaler1 = MinMaxScaler() 
        scaler2 = StandardScaler(copy=True, with_mean=True, with_std=True)
        
        # Scale feature importance
        result[imp] = result[imp].abs()
        scaled_values = scaler1.fit_transform(result[imp]) 
        scaled_values = scaler2.fit_transform(scaled_values)
        
        result[imp] = scaled_values
        
        # Reduce feature importance of RFE.
        # RFE only gives a ranking. This means that all values were one and the average was a lot higher than with the other methods
        result['Importances: Recursive Feature Elimination'] = result['Importances: Recursive Feature Elimination']/4
        
        # Create an importance score which combines the importance of all methods. Sort by total.
        result["Importance Score"] = result[imp].sum(axis=1)
        result["Importance Score"] = scaler1.fit_transform(result[["Importance Score"]])
        result = result[result["Total Count"] > threshold].sort_values(by = "Importance Score", ascending = False)
        
        return result
    
    def final_results(self, path, path_intogen, nrows=200, usecols=[x for x in range(100)], threshold=3):
        """
        - Combine: bulbasaur (from DataPrep), call_methods, add_cosmic_to_dict, results, normalize_importance.
        - Returns dataframe with results.
        """
        dataprep = magene.DataPrep(self.seed)
        feature_selection = magene.FeatureSelection(self.seed)

        X_train, y_train, X_test, y_test = dataprep.bulbasaur(path, threshold, nrows=nrows, usecols=usecols)
        dict_list = feature_selection.call_methods(X_train, y_train, X_test, y_test)
        dict_list = self.add_cosmic_to_dict(path_intogen, dict_list)
        df = self.results(dict_list)
        df = self.normalize_importance(df)
        
        return df

    def iterate_trough_cancers(self, path_list, path_intogen_list, nrows, usecols, threshold=2.5):
        """
        Main function -> iterate_through_cancers(path_list, path_intogen_list, nrows, usecols, threshold = 2.5)
        - Input: path_list (filepath to expression data chunks), path_intogen_list (filepath to cosmic data), threshold for standard deviation filter.
        - Output: Stores results as csv files.
        """
        for path, path_intogen in zip(path_list, path_intogen_list):
            start = time.time()
            
            # Create filepath for saving to csv.
            name = re.sub("^.+\/Chunk_", "", path)
            filepath = 'Output/Result_{}'.format(name)
            
            # Print beginning of loop.
            print('Evaluating {}'.format(path))      
            
            # Run models and save results to csv.
            results = self.final_results(path, path_intogen, nrows=nrows, usecols=usecols, threshold=threshold)  
            results.to_csv(filepath)
            
            # Print progress.
            print('Finished in {:.1f} min\n'.format((time.time() - start) / 60))

__all__ = ['Evaluation']