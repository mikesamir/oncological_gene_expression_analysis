import shap

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

__all__ = ['ShowImportance']