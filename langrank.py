"""
feature dataset: A dataset containing linguistic features for various languages, like the one in data/URIELPlus_Union.csv.
task dataset: A dataset containing task, transfer, and performance columns, such as the one in data/pos.csv.
"""

from typing import Tuple, List, Any
from lightgbm import LGBMRanker
import lightgbm as lgb
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import ndcg_score
from tqdm import tqdm
from urielplus.urielplus import URIELPlus
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import os
import pickle
from Latent_islands_URIEL.src.functions import distance_vector_input
from geo_distributions_helper import integrate_geo_data, normalized_w1_distance
import pkg_resources
from zipfile import ZipFile as zf
import scipy.sparse as sparse


u = URIELPlus()


class DistanceCalculator:
    """
    Base class for calculating language distances for some distance type.

    Methods
    -------
    - calculate_distance(lang1: str, lang2: str) -> float: Calculate the distance between two languages based on their linguistic features.
    """
    def __init__(self):
        self.df = None

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        """
        Calculate the distance between two languages.

        Parameters
        ----------
        - lang1: Code for language 1
        - lang2: Code for language 2

        Returns
        -------
        - Float distance between the two languages
        """
        return self._vector_distance(lang1, lang2)

    def _vector_distance(self, lang1: str, lang2: str) -> float:
        """
        Computes language distance by indexing into self.df and calculating the angular distance.

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
        if lang1 not in self.df.index:
            print(f"Language {lang1} not found in dataframe.")
            return np.nan
        if lang2 not in self.df.index:
            print(f"Language {lang2} not found in dataframe.")
            return np.nan
        idx_with_values_1 = {idx for idx, value in enumerate(self.df.loc[lang1].to_numpy()) if value != -1}
        idx_with_values_2 = {idx for idx, value in enumerate(self.df.loc[lang2].to_numpy()) if value != -1}
        intersection = list(idx_with_values_1.intersection(idx_with_values_2))
        if not intersection:
            return np.nan
        return u._angular_distance(self.df.loc[lang1].iloc[intersection].to_numpy(),
                                            self.df.loc[lang2].iloc[intersection].to_numpy())


class SyntacticCalculator(DistanceCalculator):
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the syntactic distance calculator.

        Parameters
        ----------
        - df: DataFrame containing linguistic features with syntactic columns starting with "S_"
        """
        super().__init__()
        self.df = df[[col for col in df.columns if col.startswith("S_")]]


class InventoryCalculator(DistanceCalculator):
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the inventory distance calculator.

        Parameters
        ----------
        - df: DataFrame containing linguistic features with inventory columns starting with "INV_"
        """
        super().__init__()
        self.df = df[[col for col in df.columns if col.startswith("INV_")]]


class PhonologicalCalculator(DistanceCalculator):
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the phonological distance calculator.

        Parameters
        ----------
        - df: DataFrame containing linguistic features with phonological columns starting with "P_"
        """
        super().__init__()
        self.df = df[[col for col in df.columns if col.startswith("P_")]]


class FeaturalCalculator(DistanceCalculator):
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the featural distance calculator.

        Parameters
        ----------
        - df: DataFrame containing all linguistic features
        """
        super().__init__()
        self.df = df


class MorphologicalCalculator(DistanceCalculator):
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the morphological distance calculator.

        Parameters
        ----------
        - df: DataFrame containing linguistic features with morphological columns starting with "M_"
        """
        super().__init__()
        self.df = df[[col for col in df.columns if col.startswith("M_")]]


class GenericCalculator(DistanceCalculator):
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the generic distance calculator.

        Parameters
        ----------
        - df: DataFrame containing linguistic features
        """
        super().__init__()
        self.df = df


class IslandCalculator(DistanceCalculator):
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the island distance calculator.

        Parameters
        ----------
        - df: DataFrame containing linguistic features
        """
        super().__init__()
        self.df = df

        island_path = os.path.join("Latent_islands_URIEL", "outputs", "bridged_island_full.pkl")
        with open(island_path, 'rb') as f:
            island_full = pickle.load(f)
        self.islands = island_full

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        """
        Calculate island distance between two languages.

        Parameters
        ----------
        - lang1: Code for language 1
        - lang2: Code for language 2

        Returns
        -------
        - Distance between the two languages based on island method
        """
        if self.df is None:
            raise ValueError("Dataframe is not initialized.")
        # Using imputed data, so no need to check for -1
        idx_with_values_1 = {idx for idx, value in enumerate(self.df.loc[lang1].to_numpy()) if value != -1}
        idx_with_values_2 = {idx for idx, value in enumerate(self.df.loc[lang2].to_numpy()) if value != -1}
        intersection = list(idx_with_values_1.intersection(idx_with_values_2))
        # print("shared features: ", len(intersection))
        if not intersection:
            return np.nan
        result = distance_vector_input(self.islands, self.df.loc[lang1].to_numpy(), self.df.loc[lang2].to_numpy())
        return result

class GeographicCalculator(DistanceCalculator):
    def __init__(self, language_centroid_style: int):
        """
        Initialize the geographic distance calculator.

        Parameters
        ----------
        - language_centroid_style: Style parameter for language centroid calculation
        """
        super().__init__()
        self.df = integrate_geo_data(language_centroid_style)
    
    def calculate_distance(self, lang1: str, lang2: str) -> float:
        """
        Calculate geographic distance between two languages.

        Parameters
        ----------
        - lang1: Code for language 1
        - lang2: Code for language 2

        Returns
        -------
        - Normalized geographic distance between the two languages
        """
        normalize_type = 2 #This changes the type of normalization for the final data [Can be 1 or 2]
        result = normalized_w1_distance(lang1, lang2, self.df, normalize_type)
        return result

class URIELCalculator(DistanceCalculator):
    def __init__(self, dist: str):
        """
        Initialize the URIEL distance calculator.

        Parameters
        ----------
        - dist: Distance type to use from URIEL (e.g., 'genetic', 'geographic', 'syntactic', etc.)
        """
        super().__init__()
        DISTANCES_LANGUAGE_FILE = pkg_resources.resource_filename(__name__, "l2v/l2v/data/distances_languages.txt")
        DISTANCES_FILE = pkg_resources.resource_filename(__name__, "l2v/l2v/data/distances2.zip")

        distance_to_filename = {
            "genetic": "genetic_upper_sparse.npz",
            "geographic": "geographic_upper_round1_sparse.npz",
            "syntactic": "syntactic_upper_round2_sparse.npz",
            "inventory": "inventory_upper_sparse.npz",
            "phonological": "phonological_upper_sparse.npz",
            "featural": "featural_upper_round1_sparse.npz"
        }

        def available_distance_languages():
            with open(DISTANCES_LANGUAGE_FILE) as inp:
                return inp.readline().strip().split(',')

        self.DISTANCE_LANGUAGES = available_distance_languages()

        self._cached_matrix = None

        with zf(DISTANCES_FILE, 'r') as zp:
            with zp.open(distance_to_filename[dist]) as f:
                self._cached_matrix = sparse.load_npz(f)

        self.iso_map = pd.read_csv('data/uriel_glottocode_map.csv', index_col=2).to_dict()['code']

    def calculate_distance(self, lang1: str, lang2: str) -> float:
        """
        Calculate URIEL distance between two languages.

        Parameters
        ----------
        - lang1: Code for language 1
        - lang2: Code for language 2

        Returns
        -------
        - Distance between the two languages from URIEL dataset
        """
        idx1 = self.DISTANCE_LANGUAGES.index(self.iso_map[lang1])
        idx2 = self.DISTANCE_LANGUAGES.index(self.iso_map[lang2])
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        return self._cached_matrix[idx1, idx2]

class LangRankEvaluator:
    """
    Evaluate LangRank performance in predicting transfer language based on language distances.

    Methods
    -------
    - replace_distances: Replace distances in task dataset based on newly calculated distances from calculators
    - evaluate: Run LangRank and collect NDCG@3 scores for the given task
    """
    def __init__(self, calculators: dict[str, DistanceCalculator], iso_map_file: str = 'data/code_mapping.csv'):
        """
        Initialize the LangRankEvaluator with distance calculators and an ISO map.

        Parameters
        ----------
        - calculators: Dictionary of distance calculators keyed by distance type (e.g., 'syntactic': SyntacticCalculator)
        - iso_map_file: Path to the CSV file mapping ISO codes to Glottocodes (default is 'data/code_mapping.csv')
        """
        self.calculators = calculators
        self.iso_map = pd.read_csv(iso_map_file, index_col=0).to_dict()['glottocode']

    def replace_distances(self, data_file: str, distance_types: list[str], task_col_name: str = 'task_lang', transfer_col_name: str = 'transfer_lang', iso_conversion: bool = True) -> pd.DataFrame:
        """
        Replace distances in the task dataset based on newly calculated distances from self.calculators.

        Parameters
        ----------
        - data_file: Path to the CSV file containing task, transfer, and performance columns.
        - distance_types: List of distance types to calculate (e.g., ['syntactic', 'inventory', 'phonological']). These should match the keys in self.calculators.
        - task_col_name: Name of the column containing the target language codes
        - transfer_col_name: Name of the column containing the transfer language codes
        - iso_conversion: Whether to convert language codes from ISO to Glottocode using self.iso_map

        Returns
        -------
        - DataFrame with added distance columns
        """
        if any(distance_type not in self.calculators for distance_type in distance_types):
            raise ValueError("Invalid distance types provided.")

        df = pd.read_csv(data_file)

        for index, row in tqdm(df.iterrows()):
            if iso_conversion:
                target_lang = self.iso_map[row[task_col_name]]
                transfer_lang = self.iso_map[row[transfer_col_name]]
            else:
                target_lang = row[task_col_name]
                transfer_lang = row[transfer_col_name]
            for distance_type in distance_types:
                if (distance_type == "new_geographic"):
                   code_replacements = pd.read_csv("data/glottocode_replacements.csv")
                   for idx, line in code_replacements.iterrows():
                       if (target_lang == line["bad_iso"]):
                           target_lang = line["replacement_iso"]
                       if (transfer_lang == line["bad_iso"]):
                           transfer_lang = line["replacement_iso"]
                distance = self.calculators[distance_type].calculate_distance(target_lang, transfer_lang)
                df.at[index, distance_type] = distance
        
        return df

    def evaluate(self, data: pd.DataFrame, features: list[str], performance_col_name: str, task_col_name: str = 'task_lang', transfer_col_name: str = 'transfer_lang',
                 baseline_ndcg_scores: list | None = None) -> \
    tuple[int, list[Any], float] | tuple[int, list[Any]]:
        """
        Run LangRank and collect NDCG@3 scores for the given task.

        Parameters
        ----------
        - data: DataFrame containing task, transfer, and performance columns
        - features: List of feature names to train the ranker on (e.g., ['syntactic', 'inventory', 'phonological']). These should be columns in the dataset.
        - performance_col_name: Name of the column containing the performance scores (e.g., 'f1_score')
        - task_col_name: Name of the column containing the task language codes (e.g., 'task_lang')
        - transfer_col_name: Name of the column containing the transfer language codes (e.g., 'transfer_lang')
        - baseline_ndcg_scores: Optional list of baseline NDCG scores to compare against for statistical significance testing

        Returns
        -------
        - Average NDCG@3 score across all leave-one-out iterations, individual scores, and optional p-value
        """
        if baseline_ndcg_scores is None:
            baseline_ndcg_scores = []
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
            # n_estimators=50,
            # min_child_samples=5,
            # num_leaves=8,
            # learning_rate=0.05,

            # Old parameters
            n_estimators=100,
            min_data_in_leaf=5,
            num_leaves=16,
            learning_rate=0.1,

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

        average_ndcg = np.mean(ndcg_scores) * 100

        if baseline_ndcg_scores and len(baseline_ndcg_scores) == len(ndcg_scores):
            p_value = self._statistical_significance(ndcg_scores, baseline_ndcg_scores)
            return average_ndcg, ndcg_scores, p_value

        return average_ndcg, ndcg_scores

    def _statistical_significance(self, ndcg_scores_1: list[float], ndcg_scores_2: list[float]) -> float:
        """
        Perform a paired t-test to determine if the difference between two sets of NDCG scores is statistically significant.

        Parameters
        ----------
        - ndcg_scores_1: First set of NDCG scores
        - ndcg_scores_2: Second set of NDCG scores

        Returns
        -------
        - P-value from the t-test
        """
        _, p_value = ttest_rel(ndcg_scores_1, ndcg_scores_2)
        return p_value
