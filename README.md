# Final Project DSWD-2019-03

Gene expression analysis of cancer genome data

## Project goals

Try to identify the most important protein-expressed genes that are either under or over- expressed across the different cancer types. These are potential candidate targets for new oncology drugs. Make sure that these targets are under/over-expressed in specific cell types for the corresponding cancer types and will not cause serious adverse effects (they will not cause serious issues to the normal expression in other cell types of the patient).

Data can be found via the [Xena Explorer](https://xenabrowser.net/). 

Use [this source](https://xenabrowser.net/datapages/?cohort=TCGA%20TARGET%20GTEx&removeHub=https%3A%2F%2Fxena.tree%20house.gi.ucsc.edu%3A443) to retrieve data regarding gene expressions (tpm) and phenotypes.

### Milestones 1

Identification of potential candidate targets for various cancer types.

### Milestones 2

Clustering of patients according to their gene expressions across the different cancer types. (unsupervised learning – PCA)
   
### Milestones 3

Factors that can explain difference in survival rates between clusters of patients of the same cancer type.

_References: Intelligencia.ai_

## Project File Structure

- **Data** - Storage of project data which is immutable.
- **Output** - Generated project files which should not be stored in a git repo.
- **Lu** - Lucie's notebook files for testing.
- **Mike** - Mike's notebook files for testing.
- **X_name.ipynb** - Project notebooks.