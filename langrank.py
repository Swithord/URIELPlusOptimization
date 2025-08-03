"""
feature dataset: A dataset containing linguistic features for various languages, like the one in data/URIELPlus_Union.csv.
task dataset: A dataset containing task, transfer, and performance columns, such as the one in data/pos.csv.
"""

from typing import Callable, Optional
from lightgbm import LGBMRanker
import lightgbm as lgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import ndcg_score
from urielplus.urielplus import URIELPlus
import pandas as pd
import numpy as np

class DistanceCalculator:
    """
    Language distance calculator based on linguistic features.

    Methods
    -------
    - calculate_distance(lang1: str, lang2: str) -> float: Calculate the distance between two languages based on their linguistic features.
    """
    def __init__(self, df: pd.DataFrame, column_filter: Optional[Callable[[str], bool]] = None):
        """
        Initialize the distance calculator.

        Parameters
        ----------
        - dataset_path: Path to the dataset containing linguistic features
        - column_filter: Function to filter column, follows the interface func(str)->bool. If None, uses all columns. Defaults to None.  
        """
        self.uriel = URIELPlus()
        
        if column_filter:
            self.df = df[[col for col in df.columns if column_filter(col)]]
        else:
            self.df = df

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        """
        Calculate the distance between two languages.

        Parameters
        ----------
        - lang1: Code for language 1
        - lang2: Code for language 2

        Returns
        -------
        - Angular distance between the two languages
        """
        if self.df is None:
            raise ValueError("Dataframe is not initialized.")
        
        idx_with_values_1 = {idx for idx, value in enumerate(self.df.loc[lang1].to_numpy()) if value != -1}
        idx_with_values_2 = {idx for idx, value in enumerate(self.df.loc[lang2].to_numpy()) if value != -1}
        intersection = list(idx_with_values_1.intersection(idx_with_values_2))
        
        if not intersection:
            return np.nan
            
        return self.uriel._angular_distance(
            self.df.loc[lang1].iloc[intersection].to_numpy(),
            self.df.loc[lang2].iloc[intersection].to_numpy()
        )


# Factory functions for different distance types
def create_syntactic_calculator(df: pd.DataFrame) -> DistanceCalculator:
    """Create a calculator for syntactic distances."""
    return DistanceCalculator(df, lambda col: col.startswith("S_"))

def create_inventory_calculator(df: pd.DataFrame) -> DistanceCalculator:
    """Create a calculator for inventory distances."""
    return DistanceCalculator(df, lambda col: col.startswith("INV_"))

def create_phonological_calculator(df: pd.DataFrame) -> DistanceCalculator:
    """Create a calculator for phonological distances."""
    return DistanceCalculator(df, lambda col: col.startswith("P_"))

def create_morphological_calculator(df: pd.DataFrame) -> DistanceCalculator:
    """Create a calculator for morphological distances."""
    return DistanceCalculator(df, lambda col: col.startswith("M_"))

def create_featural_calculator(df: pd.DataFrame) -> DistanceCalculator:
    """Create a calculator for featural distances (uses all features)."""
    return DistanceCalculator(df)

def create_scriptural_calculator(df: pd.DataFrame) -> DistanceCalculator:
    """Create a calculator for scriptural distances (uses all features)."""
    return DistanceCalculator(df)


class LangRankEvaluator:
    """
    Evaluates LangRank performance when using different datasets.
    """
    def __init__(self, calculators: dict[str, DistanceCalculator], iso_map_file: str = 'data/code_mapping.csv'):
        """
        Initialize the LangRankEvaluator with distance calculators and an ISO map.

        Parameters
        ----------
        - calculators: Dictionary of distance calculators keyed by distance type (e.g., 'syntactic': create_syntactic_calculator).
        - iso_map_file: Path to the CSV file mapping ISO codes to Glottocodes (default is 'data/code_mapping.csv').
        """
        self.calculators = calculators
        self.iso_map = pd.read_csv(iso_map_file, index_col=0).to_dict()['glottocode']

    def replace_distances(self, dataset_path: str, distance_types: list[str], task_col_name: str = 'task_lang', transfer_col_name: str = 'transfer_lang', iso_conversion: bool = True) -> pd.DataFrame:
        """
        Replace distances in the task dataset based on newly calculated distances from self.calculators (which are based on custom datasets).
        
        Parameters
        ----------
        - dataset_path: Path to the task dataset containing task, transfer, and performance columns (e.g. pos.csv)
        - distance_types: List of distance types to calculate (e.g., ['syntactic', 'inventory', 'phonological']).  These should match the keys in self.calculators.
        - task_col_name: Name of the column containing the target language codes
        - transfer_col_name: Name of the column containing the transfer language codes
        - iso_conversion: Whether to convert language codes from ISO to Glottocode using `iso_map_file`
        """
        if any(distance_type not in self.calculators for distance_type in distance_types):
            raise ValueError("Invalid distance types provided.")
        
        df = pd.read_csv(dataset_path)
        
        for index, row in df.iterrows():
            if iso_conversion:
                target_lang = str(self.iso_map[row[task_col_name]])
                transfer_lang = str(self.iso_map[row[transfer_col_name]])
            else:
                target_lang = str(row[task_col_name])
                transfer_lang = str(row[transfer_col_name])
            
            for distance_type in distance_types:
                distance = self.calculators[distance_type].calculate_distance(target_lang, transfer_lang)
                df.at[index, distance_type] = distance
        
        return df

    def evaluate(self, data: pd.DataFrame, features: list[str], performance_col_name: str, task_col_name: str = 'task_lang', transfer_col_name: str = 'transfer_lang') -> float:
        """
        Run LangRank and collect NDCG@3 scores for the given task.
        
        Parameters
        ----------
        - dataset_path: Path to the task dataset (containing task, transfer, and performance columns)
        - features: List of feature names to train the ranker on (e.g., ['syntactic', 'inventory', 'phonological']).
        - performance_col_name: Name of the column containing the performance scores (e.g., 'f1_score')
        - task_col_name: Name of the column containing the task language codes (default 'task_lang')
        - transfer_col_name: Name of the column containing the transfer language codes (default 'transfer_lang')

        Returns
        -------
        Average NDCG@3 score across all leave-one-out iterations
        """
        logo = LeaveOneGroupOut()
        data['relevance'] = 0

        for source_lang in data[task_col_name].unique():
            source_lang_data = data[data[task_col_name] == source_lang].copy()
            source_lang_data['rank'] = source_lang_data[performance_col_name].rank(method='min', ascending=False)
            top_indices = source_lang_data[source_lang_data['rank'] <= 10].index
            data.loc[top_indices, 'relevance'] = 11 - source_lang_data.loc[top_indices, 'rank']

        groups = data[task_col_name]
        ndcg_scores = []

        ranker = LGBMRanker(
            boosting_type='gbdt',
            objective='lambdarank',
            metric='lambdarank',

            # New parameters
            n_estimators=50,
            min_child_samples=5,
            num_leaves=8,
            learning_rate=0.05,

            # Old parameters
            # n_estimators=100,
            # min_data_in_leaf=5,
            # num_leaves=16,
            # learning_rate=0.1,

            verbose=-1
        )

        for train_idx, test_idx in logo.split(data, groups=groups):
            train_data = data.iloc[train_idx]
            test_data = data.iloc[test_idx]
            train_X = train_data[features]
            train_y = train_data['relevance']
            test_X = test_data[features]
            test_y = test_data['relevance']
            train_group_sizes = train_data.groupby(transfer_col_name).size().tolist()

            train_dataset = lgb.Dataset(train_X, label=train_y,group=train_group_sizes)
            test_dataset = lgb.Dataset(test_X, label=test_y, group=[len(test_y)], reference=train_dataset)

            ranker.fit(train_X, train_y, group=train_group_sizes, eval_group=[[len(test_y)]], eval_set=[(test_X, test_y)], eval_at=[3])
            y_pred = ranker.predict(test_X)
            ndcg = ndcg_score([test_y.values], [y_pred], k=3)
            ndcg_scores.append(ndcg)

        # lgb.plot_importance(ranker, importance_type='split')
        # plt.show()

        average_ndcg = np.mean(ndcg_scores)
        return average_ndcg * 100
