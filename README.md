# Unsupervised feature selection based on Ant Colony Optimization of the URIEL+ knowledge base
Steps:

0. `pip install -r requirements.txt`
1. Set IMPUTATION to True or False in config.py
2. Run ant_data_creation.ipynb to create the base feature set
3. Run ant.ipynb to perform the feature selection using the ant algorithm
4. Run ant_evaluation.ipynb to evaluate the selected features
5. Run ant_selection_analysis.ipynb to analyze the results of the feature selection