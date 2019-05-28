# Contains classes: FinalDataFrame, ShowImportance
# Contains functions: final_genes_expression_data

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
        import pandas as pd
        self.top_genes = pd.read_csv(top_genes_path)["top_genes"].tolist()
        self.top_genes.append("label")
        
        
    def load_data(self, chunk_path):
        """
        - Load data with bulbasaur function from DataPrep class.
        - Return X_train, y_train, X_test, y_test.
        """
        X_train, y_train, X_test, y_test = dataprep.bulbasaur(chunk_path, usecols = self.top_genes)
        return X_train, y_train, X_test, y_test
    
    
    def run_store_model(self, X_train, y_train, X_test, y_test, filename = 'Output/Models/all_cancers_model.sav'):
        """
        - Use best params from hypertuning to fit model.
        - Then store model as a pickle file.
        """
        model = XGBClassifier(colsample_bytree = 0.5,
                              gamma = 0.5,
                              learning_rate = 0.1,
                              max_depth = 5,
                              num_class = 12, 
                              n_estimators = 300, 
                              objective = "multi:softmax",
                              subsample = 0.9,
                              min_child_weight = 1)
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
        shap_values = explainer.shap_values(X_test,approximate=False)
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
        # Read top_genes data.
        top_genes = pd.read_csv(top_genes_path)

        # Merge with df_final.
        all_final = df_final.merge(top_genes, left_on = "Gene", right_on = "Top Gene")

        # Drop redundant information.
        all_final.drop(columns = "Top Gene", inplace = True)

        # Save to csv.
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
    
    
    
class ShowImportance:
    """
    Visuzalize SHAP importance
    
    Functions included:
    - importance_in_class
    - importance_across_classes
    """
    def __init__(self, loaded_model, X_test):
        """
        - Define class variables.
        """
        self.loaded_model = loaded_model
        self.X_test = X_test
        
        # Create SHAP explainer.
        self.explainer = shap.TreeExplainer(loaded_model)
        
        # Define SHAP values.
        self.shap_values = explainer.shap_values(X_test, tree_limit = -1)
        
        
    def importance_in_class(self, class_number, max_display = 10):
        """
        - Visualize most important genes for one class.
        """
        shap.summary_plot(shap_values[class_number], max_display = max_display, plot_type = "dot")
        
        
    def importance_across_classes(self, max_display = 12):
        """
        - Visualize combined importances of genes.
        - Output horizontal stacked barplot.
        """
        # Plot importances.
        shap.summary_plot(shap_values, X_test, plot_type="dot", class_names = loaded_model.classes_, max_display=max_display, layered_violin_max_num_bins=40)

        

class ExtractExpression:
    """
    Extract subset of expression data with selected genes.
    
    Functions included:
    - final_genes_expression_data.
    """
    def final_genes_expression_data(self, chunk_path, all_final_path, save_to_path):
        """
        - Extract expression data for genes that where selected only.
        - Replace Ensembl notation with gene names (ADAM33, MLANA,...)
        - Return dataframe and store as csv.
        """
        # Read all_final data (results from FinalDataFrame class).
        all_final = pd.read_csv(all_final_path)

        # Read expression data with selected genes.
        expression = pd.read_csv(chunk_path, usecols = all_final.Gene_x.tolist() + ["label", "Unnamed: 0"])
        expression.rename(columns = {"Unnamed: 0":"sample"}, inplace = True)

        # Create translation dataframe (from Ensembl to gene name).
        all_final_gene = all_final[["Gene_x", "Gene_y"]]

        ensgs = expression.columns[2:]

        # Transpose data and replace Ensembl with gene name.
        expression_T = expression.T.reset_index()
        expression_T.columns = expression_T.iloc[1]
        expression_T.drop([0,1], inplace = True)

        expression_symbol = expression_T.merge(all_final_gene, how = "outer", left_on = "label", right_on = "Gene_x").T
        expression_symbol.columns = expression_symbol.loc["Gene_y"]

        # Drop unnecessary rows
        expression_symbol.drop(["Gene_x", "Gene_y", "label"], axis = 0, inplace = True)

        # Delete duplicate columns
        expression_symbol = expression_symbol.loc[:,~expression_symbol.columns.duplicated()]

        # Save to csv.
        expression_symbol.to_csv(save_to_path)

        return expression_symbol