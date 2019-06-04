import pandas as pd

from imblearn.over_sampling import SMOTE
from boruta import BorutaPy

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet

from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.tree import export_graphviz
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.linear_model import LassoCV
from sklearn.datasets import make_regression

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve 

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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
        estimator = SVR(kernel=kernel, C=0.01, gamma=1e-07)
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
    
    def elastic_net(self, X_train_smote, y_train_res, X_test, y_test, n_features=300, alpha=0.01, l1_ratio=0.5):
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
        df = pd.concat([ft_sort.head(int(n_features/2)), ft_sort.tail(int(n_features/2))])

        # Create dictionary with results.
        selected_features = df.index.tolist()
        feature_importances = df.iloc[:,0].tolist()
        dictionary = {"Elastic Net": [selected_features, feature_importances]}

        return dictionary 
  
    def boruta_tree(self, X_train_smote, y_train_res, X_test, y_test, n_features):
        """
        - Apply Boruta two times to preselect about 400 features.
        - Decrease amount of features to n_features using a Random Forest Classifier.
        """
        # Do Boruta once as it sometimes fails with so little data twice.
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

        array_to_df = pd.DataFrame(lassoCV.coef_)
        array_to_df.index = X_train.columns
        array_sorted = array_to_df.sort_values(0, ascending=False)
        df = pd.concat([array_sorted.head(int(n_features/2)), array_sorted.tail(int(n_features/2))])
        
        # Create dictionary with results.
        feature_importances = df.iloc[:,0].tolist()
        selected_features = df.index.tolist()

        return { "Lasso CV": [selected_features, feature_importances] }
       
    def call_methods(self, X_train, y_train, X_test, y_test, n_features=300):
        """
        - Combine: rfe, gradient_boost_classifier, elastic_net, boruta_tree, lasso_cv: Lasso with crossvalidation.
        - Input: Dataset splitted to train and test size + n_features (number of features that should be selected by each method).
        - Output: Dictionaries with selected features and feature importances.
        """
        method1 = self.gradient_boost_classifier(X_train, y_train, X_test, y_test, n_features)
        method2 = self.rfe(X_train, y_train, X_test, y_test, n_features, kernel="linear")
        method3 = self.elastic_net(X_train, y_train, X_test, y_test, n_features, alpha=0.01, l1_ratio=0.5)
        method4 = self.boruta_tree(X_train, y_train, X_test, y_test, n_features)
        method5 = self.lasso_cv(X_train, y_train, X_test, y_test, n_features)

        return { **method1, **method2, **method3, **method4, **method5 }

__all__ = ['FeatureSelection']