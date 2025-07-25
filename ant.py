NUM_ANTS = 25                   # number of ants in the colony (i hate ants!!!!)
EVAPORATION = 0.2               # how much pheromone evaporates each iteration
ITERATIONS = 50                 # number of iterations to run the algorithm
FEATURES_PER_ITERATION = 200    # number of features to select per iteration
BETA = 0.5                      # weight given for similarity vs pheromone (beta > 1, the heuristic dominates)
EXPLOITATION_RATE = 0.8         # how much to exploit vs explore
EPSILON = 0.00001               # Stop division by zero errors
SIMILARITY_FUNCTION = 'phi'     # similarity function to use, 'phi' or 'mi'
SAVE_PHEROMONES = True          # whether to save pheromone matrix after each iteration
LOAD_PHEROMONES = False         # whether to load existing pheromone matrix to resume
from config import IMPUTATION   # whether to use imputed values

from urielplus import urielplus
from sklearn.metrics import matthews_corrcoef, normalized_mutual_info_score
import numpy as np
import pandas as pd
import logging
import os

from fancyimpute import SoftImpute
from langrank.replace_distances import replace_in_memory
from langrank.dep.dep import dep_in_memory
from langrank.el.el import el_in_memory
from langrank.mt.mt import mt_in_memory
from langrank.pos.pos import pos_in_memory


# cursor said this shuts up SoftImpute
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="'force_all_finite'")

# shut up LightGBM
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

 
# Initialization
logger.info("Initializing URIELPlus and integrating databases...")
uriel = urielplus.URIELPlus()
uriel.integrate_databases()
uriel.set_aggregation('U')
uriel.aggregate()
logger.info("Database integration complete!")


# Collect aggregated data
data: np.ndarray = np.squeeze(uriel.get_typological_data_array())
logger.info(f"Data shape: {data.shape}")
 
# Similarity Functions
def phi_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the absolute value of Matthews Correlation Coefficient (phi coefficient) for two binary vectors.

    Input
    -----
    - x: array-like (num_features,)
    - y: array-like (num_features,)

    Output
    ------
    The absolute value of the phi coefficient
    """
    return abs(matthews_corrcoef(x, y))


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the normalized mutual information between two binary variables.

    Input
    -----
    - x: array-like (num_features,)
    - y: array-like (num_features,)

    Output
    ------
    The normalize mutual information score between x and y.
    """
    return normalized_mutual_info_score(x, y)

 
# Helper Functions
def construct_weight_matrix(data: np.ndarray, similarity_function) -> np.ndarray:
    """
    Construct a weight (similarity) matrix from the data.

    Input
    -----
    - data: array-like (num_samples, num_features)
    - similarity_function: function to compute similarity between two features, follows the signature; func([ndarray], [ndarray]) -> float
    
    Output
    ------
    Square weight matrix of shape (num_features, num_features) where Wij is the similarity between feature i and feature j. 
    Wii is set to the maximum similarity value.
    """
    num_features = data.shape[1]
    weights = np.ones((num_features, num_features)) # Might not be 1 for MI and average aggregation
    
    for i in range(num_features):
        for j in range(i):
            weights[i, j] = similarity_function(data[:, i], data[:, j])
            weights[j, i] = weights[i, j]
    
    return weights

def construct_pheromone_matrix(data: np.ndarray) -> np.ndarray:
    """
    Construct a pheromone matrix from the data.

    Input
    -----
    - data: array-like, shape (n_samples, n_features)

    Output
    ------
    - pheromones: array-like, shape (n_features,)
    """
    num_features = data.shape[1]
    
    # Try to load existing pheromones if requested
    pheromone_file = f'pheromones_{SIMILARITY_FUNCTION}_{NUM_ANTS}_{FEATURES_PER_ITERATION}.npy'
    if LOAD_PHEROMONES:
        try:
            pheromones = np.load(pheromone_file)
            logger.info(f"Loaded existing pheromone matrix from {pheromone_file}")
            return pheromones
        except FileNotFoundError:
            logger.info(f"No existing pheromone file found at {pheromone_file}, starting fresh")
    
    pheromones = np.ones((num_features,))
    
    return pheromones

def compute_pheromones(data: np.ndarray, selected_features: set[int]) -> float:
    """
    Compute the pheromones for a given subset of features evaluated on LangRank.

    Input
    -----
    - data: array-like, shape (n_samples, n_features), the full URIEL dataset before feature selection
    - subset: set of feature indices that are selected

    Output
    ------
    - loss: float, the loss value computed from the LangRank results
    """
    FEATURE_TYPES = ['GENETIC','SYNTACTIC','FEATURAL','PHONOLOGICAL','INVENTORY','GEOGRAPHIC']

    subset: np.ndarray = data[:, list(selected_features)]

    if IMPUTATION:
        subset = np.where(subset == -1, np.nan, subset)
        imputer = SoftImpute(max_iters=400,  max_value=1, min_value=0, init_fill_method="mean", verbose=False)
        imputed_values = imputer.fit_transform(subset)

        df = pd.DataFrame(imputed_values, columns=uriel.get_typological_features_array()[np.array(list(selected_features))], index=uriel.get_typological_languages_array())
    else:
        df = pd.DataFrame(subset, columns=uriel.get_typological_features_array()[np.array(list(selected_features))], index=uriel.get_typological_languages_array())

    dep_df, el_df, mt_df, pos_df = replace_in_memory(df)
    dep_ndcg: float = dep_in_memory(dep_df, FEATURE_TYPES)
    el_ndcg: float = el_in_memory(el_df, FEATURE_TYPES)
    mt_ndcg: float = mt_in_memory(mt_df, FEATURE_TYPES)
    pos_ndcg: float = pos_in_memory(pos_df, FEATURE_TYPES)

    reward: float = float(np.mean([dep_ndcg, el_ndcg, mt_ndcg, pos_ndcg]))

    return reward
 
# Unsupervised Feature Selection based on Ant Colony Optimization (UFSACO)
# Based on "An unsupervised feature selection algorithm based on ant colony optimization" (Tabakhi, 2014).
class Ant:
    """
    Class representing an ant in the Ant Colony Optimization algorithm.
    Each ant has a current feature and a set of selected features.
    """
    def __init__(self, initial_feature: int):
        self.current_feature = initial_feature
        self.selected_features: set[int] = set()


def choose_feature(ant: Ant, pheromones: np.ndarray, weights: np.ndarray, mode: str = 'prob') -> int:
    """
    Choose a feature for the ant to select based on pheromone levels and weights.
    
    Inputs
    ------
    - ant: Ant object representing the current ant
    - pheromones: array-like, shape (n_features,)
    - weights: array-like, shape (n_features, n_features)
    - mode: str, either 'prob' for probabilistic selection or 'greedy' for deterministic selection - default 'prob'

    Output
    ------
    The index of the selected feature.
    """
    num_features = pheromones.shape[0]

    if len(ant.selected_features) == num_features:
        raise ValueError("All features have been selected. Ensure that FEATURES_PER_ITERATION is less than the number of features.")

    mask = np.zeros(num_features, dtype=int)
    mask[list(ant.selected_features)] = 1

    logits = pheromones * ((1/(weights[ant.current_feature, :] + EPSILON)) ** BETA)
    logits = np.where(mask == 0, logits, 0)

    if mode == 'prob': 
        probabilities: np.ndarray = logits / np.sum(logits)
        feature: int = np.random.choice(range(num_features), p=probabilities)
    elif mode == 'greedy':        
        feature: int = int(np.argmax(logits))
    else:
        raise ValueError("Invalid mode. Choose 'prob' or 'greedy'.")

    return feature


def select_features_ACO(data: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Select features using the Ant Colony Optimization algorithm.
    The algorithm iteratively updates pheromone levels based on the features selected by the ants.

    Inputs
    ------
    - data: array-like, shape (n_languages, n_features)
    - weights: array-like, shape (n_features, n_features), the similarity matrix between features

    Outputs
    -------
    - ndarray of indicies of features, ordered by final pheromone levels in descending order.
    - ndarray of pheromone levels, ordered by final pheromone levels in descending order.
    """
    num_features = data.shape[1]

    pheromones: np.ndarray = construct_pheromone_matrix(data)

    for iteration in range(ITERATIONS):
        # Initial placement of ants (no duplicates), does not count towards feature_count or pheromones
        ants: list[Ant] = []
        placed_features = set()
        for _ in range(NUM_ANTS):
            feature: int = np.random.choice(list(set(range(num_features)) - placed_features))
            ant = Ant(feature)
            placed_features.add(feature)
            ants.append(ant)

        # Iterate
        for feature_choice in range(FEATURES_PER_ITERATION):
            for ant in ants:
                mode: str = 'prob' if np.random.rand() > EXPLOITATION_RATE else 'greedy'
                feature: int = choose_feature(ant, pheromones, weights, mode)
                ant.current_feature = feature
                ant.selected_features.add(feature)

        # Update pheromones
        pheromones *= (1 - EVAPORATION)
        for ant in ants:
            reward = compute_pheromones(data, ant.selected_features)
            pheromones[np.array(sorted(ant.selected_features), dtype=int)] += max(0, reward)

        logger.info(f"Iteration {iteration + 1}/{ITERATIONS} - Best pheromone: {np.max(pheromones):.4f}")
        
        # Save pheromones after each iteration if requested
        if SAVE_PHEROMONES:
            pheromone_file = f'pheromones_{SIMILARITY_FUNCTION}_{NUM_ANTS}_{FEATURES_PER_ITERATION}.npy'
            np.save(pheromone_file, pheromones)
            logger.info(f"Saved pheromone matrix to {pheromone_file}")

    # Sort features by pheromone levels in descending order
    return np.argsort(pheromones)[::-1], np.sort(pheromones)[::-1]

 
# Prepare data
data: np.ndarray = np.squeeze(uriel.get_typological_data_array())
feature_labels: np.ndarray = uriel.get_typological_features_array()
languages: np.ndarray = uriel.get_typological_languages_array()
df = pd.DataFrame(data, columns=feature_labels, index=languages)

# Run ACO algorithm
similarity_function = phi_coefficient if SIMILARITY_FUNCTION == 'phi' else mutual_information
logger.info("Constructing weight matrix...")
weights: np.ndarray = construct_weight_matrix(data, similarity_function)
logger.info("Starting feature selection with ACO...")
ranked_features, pheromones = select_features_ACO(data, weights)
logger.info("ACO feature selection complete!")

# Save results
logger.info("Saving results...")
for num_features in range(100, 701, 100):
    logger.info(f"Processing {num_features} features...")
    filtered_data: pd.Series = df.iloc[:, ranked_features[:num_features]]

    df_np = filtered_data.to_numpy()
    
    df_final = pd.DataFrame(df_np, columns=filtered_data.columns, index=filtered_data.index)

    if not os.path.exists('selection_result'):
        os.makedirs('selection_result')

    if not os.path.exists('eval_result'):
        os.makedirs('eval_result')

    df_final.to_csv(f'selection_result/ant_{SIMILARITY_FUNCTION}_{num_features}.csv')
    logger.info(f"Saved: {f'selection_result/ant_{SIMILARITY_FUNCTION}_{num_features}.csv'}")

logger.info("All processing complete!")