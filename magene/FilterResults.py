import pandas as pd

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
    
    def top_all(self, dictionary, chunks):
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
        top_all_df = self.top_all(dictionary, chunk_names)
        top_genes_info = self.atlas_info(top_all_df)
        
        return top_genes_info

__all__ = ['FilterResults']