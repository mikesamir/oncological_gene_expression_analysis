import pandas as pd
import numpy as np
import pickle
import shap
import magene

class FinalDataFrame:
    """
    Apply XGBoost to get feature importance across classes and combine with results from GeneSelection.
    
    Main function -> squirtle(self, chunk_path, pickle, top_genes_path, save_to_path, top_genes_n = 20):
    - Input: 
        chunk_path (all cancers data chunk)
        pickle (path for pickle file of XGBoost model) 
        top_genes_path (results from GeneSelection library)
        save_to_path (where to save)
        top_genes_n (how many of the top genes of each tissue type from SHAP should be used)
    - Output:
        Dataframe with all information about the genes.
    
    Functions included:
    - load_data
    - run_store_model
    - load_model
    - get_genes_df
    - add_combination
    - combine_ith_top_genes
    """
    def __init__(self, seed, top_genes_path):
        self.seed = seed
        self.top_genes_path = top_genes_path
        self.top_genes = pd.read_csv(top_genes_path)["Top Gene"].tolist()
        self.top_genes.append("label")
        
    def load_data(self, chunk_path):
        """
        - Load data with bulbasaur function from DataPrep class.
        - Return X_train, y_train, X_test, y_test.
        """
        dataprep = magene.DataPrep(self.seed)

        X_train, y_train, X_test, y_test = dataprep.bulbasaur(chunk_path, usecols=self.top_genes)
        return X_train, y_train, X_test, y_test

    def run_store_model(self, X_train, y_train, X_test, y_test, filename):
        """
        - Use best params from hypertuning to fit model.
        - Then store model as a pickle file.
        """
        model = XGBClassifier(colsample_bytree=0.5,
                              gamma=0.5,
                              learning_rate=0.1,
                              max_depth=5,
                              num_class=12, 
                              n_estimators=300, 
                              objective="multi:softmax",
                              subsample=0.9,
                              min_child_weight=1)
        model.fit(X_train, y_train)
        pickle.dump(model, open(filename, 'wb')) 
        
    def load_model(self, filename):
        # load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        
        return loaded_model
    
    def get_genes_df(self, X_train, loaded_model, top_genes_n):
        """
        Get top_genes dataframe with corresponding SHAP values for all cancers
        """
        explainer = shap.TreeExplainer(loaded_model)
        X_importance = X_train
        shap_values = explainer.shap_values(self.X_test, approximate=False)
        importance_all = pd.DataFrame()

        for i, cancer in enumerate(loaded_model.classes_):
            #print(i, cancer)

            shap_sum = np.abs(shap_values[i]).mean(axis=0)
            importance_df = pd.DataFrame([X_importance.columns.tolist(), shap_sum.tolist()]).T
            importance_df.columns = ['Gene', 'Shap_Importance']
            importance_df = importance_df.sort_values('Shap_Importance', ascending=False).head(top_genes_n)
            importance_df["Cancer"] = cancer
            importance_all = importance_all.append(importance_df)
            
        return importance_all

    def add_combination(self, df):
        """
        Add a column which gives the combination of cancers for duplicated genes
        """
        df_final = pd.DataFrame()
        for gene in df["Gene"].unique():
            df1 = df[df["Gene"] == gene].copy()
            df1["Combination"] = df1["Cancer"].str.cat(sep = ", ")
            df1["SHAP_Combination"] = df1["Shap_Importance"].astype("str").str.cat(sep = ", ")
            df_final = df_final.append(df1)
            
        return df_final
    
    def combine_with_top_genes(self, df_final, top_genes_path, save_to_path):
        """
        - Combine results of XGBoost with information from the other models and Protein Atlas Information.
        - Returns all_final, a final representation of the results.
        """
        top_genes = pd.read_csv(top_genes_path)
        all_final = df_final.merge(top_genes, left_on = "Gene", right_on = "Top Gene")
        all_final.drop(columns = "Top Gene", inplace = True)
        all_final.to_csv(save_to_path)
        
    def squirtle(self, chunk_path, pickle, top_genes_path, save_to_path, top_genes_n = 20):
        """
        Combine: 
        Input: Chunk_path
        Output: Dataframe with shap importance and combinations (Returns and saves as csv.).
        """
        self.X_train, self.y_train, self.X_test, self.y_test = self.load_data(chunk_path)
        loaded_model = self.load_model(pickle)
        df = self.get_genes_df(self.X_train, loaded_model, top_genes_n)
        df_final = self.add_combination(df)
        all_final = self.combine_with_top_genes(df_final, top_genes_path, save_to_path)
        
        return df_final

__all__ = ['FinalDataFrame']