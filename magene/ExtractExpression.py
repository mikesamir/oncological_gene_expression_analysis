import pandas as pd

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

__all__ = ['ExtractExpression']