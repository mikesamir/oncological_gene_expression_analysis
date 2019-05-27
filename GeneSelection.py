# Contains classes: FeatureSelection, Evaluation, FilterResults
    
class FeatureSelection:
    """
    Applies various feature selection algorithms and stores results (features + feature importance) in a dictionary.
    
    Main function -> call_methods(X_train, y_train, X_test, y_test, n_features = 300)
    - Input: Dataset splitted to train and test size + n_features (number of features that should be selected by each method).
    - Output: Dictionaries with selected features and feature importances.

    Included functions:
    - rfe: Recursive Feature Elimination.
    - gradient_boost_classifier.
    - elastic_net.
    - boruta_tree: Boruta with Random Forest Classifier at the end.
    - lasso_cv: Lasso with crossvalidation.
    """
    def __init__(self, seed):
        self.seed = seed
    
    
    def rfe(self, X_train, y_train, X_test, y_test, n_features = 300, step = 0.2, kernel = "linear"):
        """
        - Recursive Feature Elimination - step < 1 is a percentage. Returns selected features.
        - Hyperparameter tuning was done in a different notebook.
        """
        # Create estimator and selector.
        estimator = SVR(kernel=kernel, C = 0.01, gamma = 1e-07)
        selector = RFE(estimator, n_features_to_select = n_features, step=step)
        selector = selector.fit(X_train.to_numpy(), y_train.to_numpy())
        
        # Print accuracy.
        print('Accuracy of RFE: {:.3f}'.format(selector.score(X_test, y_test)))
        
        # Create dictionary with results.
        selected_features = X_train.columns[selector.support_].tolist()
        feature_importances = [1 for x in range(len(selected_features))]
        dictionary = {"Recursive Feature Elimination":[selected_features, feature_importances]}
        
        return dictionary
  
  
    def gradient_boost_classifier(self, X_train, y_train, X_test, y_test, n_features = 300):
        """
        Gradient Boost Classifier with feature importance selection.
        Hypertuned parameters:
        {'learning_rate': 0.5, 'max_depth': 3, 'min_samples_leaf': 7, 'min_samples_split': 0.1, 'n_estimators': 100, 'random_state': 1888, 'subsample': 0.75}
        """
        # Create Gradient Boost Classifier.
        new = GradientBoostingClassifier(learning_rate=0.5, n_estimators=100, max_depth=3,
                                         min_samples_split=0.1, min_samples_leaf=7,\
                                         subsample=0.75, random_state=self.seed)
        new.fit(X_train,y_train)
        predictors = list(X_train)
        
        # Retrieve feature importance.
        feat_imp = pd.Series(new.feature_importances_, predictors).sort_values(ascending=False)[:n_features]
        
        pred = new.predict(X_test)
        
        # Print accuracy.
        print('Accuracy of GBM: {:.3f}'.format(new.score(X_test, y_test)))
    
        # Add features and feature importance to dictionary.
        importances = new.feature_importances_
        genes = X_test.columns
        
        # Create dictionary with results.
        selected_features_df = pd.DataFrame(importances, index = genes).sort_values(0, ascending = False).head(n_features)
        selected_features = selected_features_df.index.tolist()
        feature_importances = selected_features_df.iloc[:,0].tolist()
        dictionary = {"Gradient Boost Classifier":[selected_features, feature_importances]}
        
        return dictionary
    
    
    def elastic_net(self, X_train_smote, y_train_res, X_test, y_test, alpha=0.01, l1_ratio=0.5, n_features=300):
        """
        - Elastic net with feature importance
        """
        # Define selector.
        clf = ElasticNet(random_state=self.seed, alpha=alpha, l1_ratio=l1_ratio)
        clf.fit(X_train_smote, y_train_res)
        clf.pred = clf.predict(X_test)
        
        # Print accuracy.
        print("Accuracy of Elastic Net: {:.3f}".format(clf.score(X_test, y_test)))
        
        # Retrieve feature importance.
        ft_imp = pd.DataFrame(clf.coef_, index=X_train_smote.columns)
        ft_sort = ft_imp.sort_values(0, ascending=False)
        imp_coef = pd.concat([ft_sort.head(int(n_features/2)), ft_sort.tail(int(n_features/2))])

        # Create dictionary with results.
        selected_features = imp_coef.index.tolist()
        feature_importances = imp_coef.iloc[:,0].tolist()
        dictionary = {"Elastic Net": [selected_features, feature_importances]}

        return dictionary 
  
    def boruta_tree(self, X_train_smote, y_train_res, X_test, y_test, n_features):
        """
        - Apply Boruta two times to preselect about 400 features.
        - Decrease amount of features to n_features using a Random Forest Classifier.
        """
        # Do Boruta twice.
        for _ in range(1):

            from sklearn.metrics import f1_score # import again to avoid error...

            # Random Forests for Boruta.
            rf_boruta = RandomForestClassifier(n_jobs=-1, random_state=self.seed)

            # Perform Boruta.
            boruta = BorutaPy(rf_boruta, n_estimators='auto', verbose=0,
                          alpha=0.005, max_iter=30, perc=100, random_state=self.seed)
            boruta.fit(X_train_smote.values, y_train_res)

            # Select features and fit Logistic Regression.
            cols = X_train_smote.columns[boruta.support_]
            X_train_smote = X_train_smote[cols]
            est_boruta = LogisticRegression(random_state=self.seed)
            est_boruta.fit(X_train_smote, y_train_res)

            scores = cross_val_score(est_boruta, X_train_smote, y_train_res, cv=5)
            
            # Print accuracy.
            print("Accuracy of Boruta: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

        # Random Forest for extracting features.
        X_filtered = X_train_smote[cols]
        
        # Define selector.
        rf = RandomForestClassifier(n_estimators = 10, criterion = 'gini', random_state = self.seed)
        rf.fit(X_filtered, y_train_res)
        rf_pred = rf.predict(X_test[cols])
        
        # Print accuracy.
        print("Accuracy of Boruta Tree: {:.3f}".format(accuracy_score(y_test, rf_pred)))
        
        # Retrieve features and importance.
        feature_names = X_filtered.columns
        rf_coeff = pd.DataFrame({"feature": feature_names,"coefficient": rf.feature_importances_})
        rf_coeff_top = rf_coeff.sort_values(by="coefficient",ascending=False).head(n_features).set_index("feature")
        
        # Create dictionary with results.
        selected_features = rf_coeff_top.index.tolist()
        feature_importances = rf_coeff_top.coefficient.tolist()
        dictionary = {"Boruta Tree": [selected_features, feature_importances]}

        return dictionary
    
    def lasso_cv(self, X_train, y_train, X_test, y_test, n_features = 300):
        """
        - Feature selection through lasso with cross-validation.
        """
        # Define LassoCV.
        lassoCV = LassoCV(cv=3, random_state=1888).fit(X_train, y_train)
        
        # Print accuracy.
        print("Accuracy of LassoCV {:.3f}".format(lassoCV.score(X_test, y_test)))

        def imp_coef(model, n, columns=X_train.columns):
            """
            - Retrieve feature importance from model.
            """
            array_to_df = pd.DataFrame(model.coef_)
            array_to_df.index = columns
            array_sorted = array_to_df.sort_values(0, ascending=False)
            imp_coef = pd.concat([array_sorted.head(int(n/2)), array_sorted.tail(int(n/2))])
            
            # Create dictionary with results.
            feature_importances = imp_coef.iloc[:,0].tolist()
            selected_features = imp_coef.index.tolist()
            dictionary = {"Lasso CV": [selected_features, feature_importances]}
            
            return dictionary
        
        # Call imp_coef function to create dictionary.
        dict_ = imp_coef(lassoCV, n = n_features)
        
        return dict_
        
        
    def call_methods(self, X_train, y_train, X_test, y_test, n_features = 300):
        """
        - Combine: rfe, gradient_boost_classifier, elastic_net, boruta_tree, lasso_cv: Lasso with crossvalidation.
        - Input: Dataset splitted to train and test size + n_features (number of features that should be selected by each method).
        - Output: Dictionaries with selected features and feature importances.
        """
        method1 = self.gradient_boost_classifier(X_train, y_train, X_test, y_test, n_features)
        method2 = self.rfe(X_train, y_train, X_test, y_test, n_features, kernel = "linear")
        method3 = self.elastic_net(X_train, y_train, X_test, y_test, n_features, alpha=0.01, l1_ratio=0.5)
        method4 = self.boruta_tree(X_train, y_train, X_test, y_test, n_features)
        method5 = self.lasso_cv(X_train, y_train, X_test, y_test, n_features)

        return { **method1, **method2, **method3, **method4, **method5}
    
    
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
    def add_cosmic_to_dict(self, path, dict_list):
        """
        - Add a dictionary with Cosmic cancer-related genes and corresponding Mutation Count.
        """
        #path = "Data/Intogen_Data/Lung_Adenocarcinoma_LUAD_TCGA.tsv"
        census = pd.read_csv("Data/Reference_Data/Census_allWed May 15 09_46_55 2019.csv")
        census["GENE"] = census["Synonyms"].str.extract(pat = '(ENSG...........)')
        census = census[census["GENE"].notnull()]
        census.fillna("None", inplace = True)
        importances = census["Role in Cancer"].tolist()
        cosmic_genes = {"Cosmic":[census["GENE"].to_list(), importances]}
        dict_list = {**dict_list, **cosmic_genes}
        
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
    
    
    def final_results(self, path, path_intogen, nrows = 200, usecols = [x for x in range(100)], threshold = 3):
        """
        - Combine: bulbasaur (from DataPrep), call_methods, add_cosmic_to_dict, results, normalize_importance.
        - Returns dataframe with results.
        """
        X_train, y_train, X_test, y_test = dataprep.bulbasaur(path, threshold, nrows = nrows, usecols = usecols)
        dict_list = FS.call_methods(X_train, y_train, X_test, y_test)
        dict_list = evaluation.add_cosmic_to_dict(path_intogen, dict_list)
        df = evaluation.results(dict_list)
        df = evaluation.normalize_importance(df)
        
        return df

    
    def iterate_trough_cancers(self, path_list, path_intogen_list, nrows, usecols, threshold = 2.5):
        """
        Main function -> iterate_through_cancers(path_list, path_intogen_list, nrows, usecols, threshold = 2.5)
        - Input: path_list (filepath to expression data chunks), path_intogen_list (filepath to cosmic data), threshold for standard deviation filter.
        - Output: Stores results as csv files.
        """
        for path, path_intogen in zip(path_list, path_intogen_list):
            start = time.time()
            
            # Create filepath for saving to csv.
            name = re.sub("^.+\/Chunk_", "2.0_", path)
            filepath = 'Output/Results/Result_{}'.format(name)
            
            # Print beginning of loop.
            print('Evaluating {}'.format(path))      
            
            # Run models and save results to csv.
            results = evaluation.final_results(path, path_intogen, nrows = nrows, usecols = usecols, threshold = threshold)  
            results.to_csv(filepath)
            
            # Print progress.
            print('Finished in {:.1f} min\n'.format((time.time() - start) / 60))


class FilterResults:
    """
    Filter for best genes based on different criteria (Total Count, Importance, Overlaps with public data).
    
    Main function -> charmander(results_list, chunk_names):
    - input: chunk_names (names of cancer tissues), results_list (list of results dataframes created by Evaluation class).
    - ouput: Dataframe with selected genes and additional information from Human Protein Atlas.

    Included functions:
    - top: Retrieve top genes (based on 1. Importance Score, 2. Total Count, 3. Overlaps with Cosmic Genes).
    - top_all: Combines top genes from all cancer types.
    - atlas_info: Add Human Protein Atlas information.
    """
    def __init__(self, seed):
        """
        Initialize seed
        """
        self.seed = seed

        
    def top(self, results_list, chunk_names):
        """
        This function selects the top 50 genes
        The selection is based on:
        - Importance Score
        - Total Count
        - Overlaps with Cosmic
        """
        dictionary = {}
        for results, chunk_name in zip(results_list, chunk_names):
            # Top genes
            top = []
            # Chosen based on Count (C), Importance Score (I) or Cosmic Overlap (O).
            top_criteria = []

            top_30_count = [results.sort_values(by="Total Count", ascending = False).head(30).index.tolist(), ["T"]*30]
            top_30_score = [results.sort_values(by="Importance Score", ascending = False).head(30).index.tolist(), ["I"]*30]

            results = results[results["Total Count"] > 1]
            cosmic_genes = results[results["Cosmic"] == 1].index.tolist()[:30]
            top_30_cosmic = [cosmic_genes, ["C"]*len(cosmic_genes)]

            for lst in [top_30_count, top_30_score, top_30_cosmic]:
                for element in lst[0]:
                    top.append(element)
                for element in lst[1]:
                    top_criteria.append(element)
            top_df = pd.DataFrame({"Gene":top, "Criteria":top_criteria})

            # Create Dataframe with duplicate genes removed and remaining labelled according to matching criteria.
            seen = set()
            top_unique = [x for x in top if x not in seen and not seen.add(x)]
            top_criteria_code = []

            for gene in top_unique:
                top_criteria_code.append(top_df[top_df["Gene"] == gene]["Criteria"].sum())

            dictionary[chunk_name] = [top_unique, top_criteria_code]
            
            top_unique_df = pd.DataFrame({"Gene":top_unique, "Criteria":top_criteria_code})
            print(chunk_name, ": ", str(len(top_unique)), "\tgenes selected |", str(len(top)-len(top_unique)), "duplicates removed",
                 "\tT:", str(top_unique_df["Gene"][top_unique_df["Criteria"] == "T"].count()),
                 ",I:", str(top_unique_df["Gene"][top_unique_df["Criteria"] == "I"].count()),
                 ",C:", str(top_unique_df["Gene"][top_unique_df["Criteria"] == "C"].count()),
                 ",TI:", str(top_unique_df["Gene"][top_unique_df["Criteria"] == "TI"].count()),
                 ",TC:", str(top_unique_df["Gene"][top_unique_df["Criteria"] == "TC"].count()),
                 ",IC:", str(top_unique_df["Gene"][top_unique_df["Criteria"] == "IC"].count()),
                 ",TIC:", str(top_unique_df["Gene"][top_unique_df["Criteria"] == "TIC"].count()),)
            
        return dictionary
    
    
    def top_all(self, dictionary):
        """
        - Combine top genes of each cancer chunk in one dataframe.
        - When duplicated the criteria will be concatenated.
        """
        # Create skeleton of dataframe.
        top_all = []
        top_all_criteria = []
        for chunk in chunks:
            for element in dictionary[chunk][0]:
                top_all.append(element)
            for element in dictionary[chunk][1]:
                top_all_criteria.append(element)
        top_all_df = pd.DataFrame({"Gene":top_all, "Criteria":top_all_criteria})
        
        # Extract unique genes.
        seen = set()
        top_all_unique = [x for x in top_all if x not in seen and not seen.add(x)]
        top_all_criteria_code = []
        
        # Concatenate criteria code when duplicated.
        for gene in top_all_unique:
            top_all_criteria_code.append(top_all_df[top_all_df["Gene"] == gene]["Criteria"].sum())
                
        print("Top genes overall: ", str(len(top_all_unique)), "genes selected |", str(len(top_all)-len(top_all_unique)), "duplicates removed")
        
        # Create dataframe.
        top_all_df = pd.DataFrame({"Top Gene":top_all_unique, 
                             "Criteria":top_all_criteria_code, 
                             "Number of picks":[len(x) for x in top_all_criteria_code]})

        return top_all_df.sort_values(by = "Number of picks", ascending = False)
       
    
    def atlas_info(self, top_genes_df):
        """
        - Incorporate Human Protein Atlas information (protein description, location in genome, ...).
        - Data retrieved from https://www.proteinatlas.org/.
        """
        # Load atlas data.
        atlas = pd.read_csv("Data/proteinatlas.tsv", sep = "\t")
        
        # Merge atlas data with top_genes dataframe.
        top_genes_info = top_genes_df.merge(atlas, left_on = "Top Gene", right_on = "Ensembl")
        top_genes_info = top_genes_info.set_index("Top Gene").drop(columns = "Ensembl")

        return top_genes_info
    
    
    def charmander(self, results_list, chunk_names):
        """
        - Combine: top, top_all, atlas_info.
        - input: chunk_names (names of cancer tissues), results_list (list of results dataframes created by Evaluation class).
        - ouput: Dataframe with selected genes and additional information from Human Protein Atlas.
        """
        dictionary = self.top(results_list, chunk_names)
        top_all_df = self.top_all(dictionary)
        top_genes_info = self.atlas_info(top_all_df)
        
        return top_genes_info
