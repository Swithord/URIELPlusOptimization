# Set if using imputed values
IMPUTATION = False

# The proper way to use these files are:
# 1. Set IMPUTATION to True or False in config.py
# 2. Run ant_data_creation.ipynb to create the base feature set
# 3. Run ant.ipynb to perform the feature selection using the ant algorithm
# 4. Run ant_evaluation.ipynb to evaluate the selected features
# 5. Run ant_selection_analysis.ipynb to analyze the results of the feature selection