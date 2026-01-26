# Input: LMSYS 25k csv file from Su & Aarzoo
# Output: clustering_params.Data format

# from typing import Sequence
from absl import app
from absl import flags
import numpy as np
import json

from clustering import clustering_algorithm
from clustering import clustering_params
from clustering.demo import data_generation

"""Dataset and metadata needed by the clustering algorithm.

  Attributes:
    datapoints: Datapoints where each row is a datapoint.
    radius: Bound on the distance of each point from the origin in datapoints.
    labels: Labels, if any, for the data (None by default).
    num_points: Number of datapoints, populated based on the number of rows in
      datapoints (inferred from datapoints).
    dim: Dimension of each datapoint, populated based on the number of columns
      in datapoints (inferred from datapoints).
  
  datapoints: Points
  radius: float
  labels: Optional[np.ndarray] = None
  num_points: int = dataclasses.field(init=False)
  dim: int = dataclasses.field(init=False)
"""

FLAGS = flags.FLAGS

_FIXED_EPS = flags.DEFINE_float(
    'fixed_epsilon', 1.0,
    'Value of epsilon to use when experimenting with varying k.')
_K_TO_TRY = flags.DEFINE_list(
    'k_to_try', '2, 4, 8, 16, 32, 64',
    'List of k values to use when experimenting with varying k.')
_FIXED_K = flags.DEFINE_integer(
    'fixed_k', 64, 'Value of k when experimenting with varying epsilon.')
_EPS_TO_TRY = flags.DEFINE_list(
    'epsilon_to_try', '0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, inf',
    'List of epsilon values to use when experimenting with varying epsilon.')

# Q. how to determine radius for clip_to_radius()?
# A. I don't think we need a radius (this is for DP clustering sensitivity issues
# but we don't even compute cluster centers i.e., weighted averages)
# A2. I think we do need clipping as a preprocessing step so data will have DP privacy.
# But what is the precise DP privacy guarantee based on radius choice?

input_json = "/Users/ihyunnam/Downloads/summary_embeddings.json"

def prep_data() -> clustering_params.Data:
    radius = 1.0   # TODO experiment (untouched norms.min()=0.99999, norms.max()=1.0000001 after previous normalization)

    with open(input_json, "r") as f:
        rawdata = json.load(f)

    EMBEDDING_KEY = "embeddings"

    datapoints = np.array(
        rawdata[EMBEDDING_KEY],
        dtype=np.float32
    )

    data = clustering_params.Data(datapoints, radius)
    data_clipped = data.clip_by_radius()
    
    # SANITY CHECK: max ~= radius 
    norms = np.linalg.norm(data_clipped, axis=1)
    print(norms.min(), norms.max())

    return clustering_params.Data(data_clipped, radius)

eval_head = ('|  k | epsilon | clustering loss |    dominant label accuracy '
               '   | false match fraction | true non-match fraction |')

def run_clustering(k: int, eps: float, data: clustering_params.Data) -> None:
  privacy_param = clustering_params.DifferentialPrivacyParam(
      epsilon=eps, delta=1e-6)
  tree_param = clustering_params.TreeParam(
      min_num_points_in_branching_node=200,
      min_num_points_in_node=100,
      max_depth=5,
  )
  clustering_result: clustering_algorithm.ClusteringResult = (
      clustering_algorithm.private_lsh_clustering(
          k,
          data,
          privacy_param,
          tree_param=tree_param))

  norm_loss = clustering_result.loss / data.num_points
  print(f"clustering result loss: {norm_loss}")
#   clustering_metrics: clustering_algorithm.ClusteringMetrics = (
#       clustering_result.get_clustering_metrics())
#   correct_pred = clustering_metrics.dominant_label_correct_count
#   accuracy = clustering_metrics.dominant_label_accuracy
#   false_match_frac = clustering_metrics.false_match_frac
#   true_nonmatch_frac = clustering_metrics.true_nonmatch_frac
#   print(
#       f'| {k:>2} | {eps:>7} '
#       f'| {clustering_result.loss:>15.8} '
#       f'| {accuracy:>6.2} ({correct_pred:>6} out of {_NUM_POINTS.value:>6}) '
#       f'| {false_match_frac:>20.4} '
#       f'| {true_nonmatch_frac:>23.4} |')

def main(argv):
    del argv  # unused
    print("Hello")
    datapoints = prep_data()

    print(f'\n# Evaluation with epsilon = {_FIXED_EPS.value} and '
          f'varying k in {list(map(int, _K_TO_TRY.value))}')
    print(eval_head)
    # for k in list(map(int, _K_TO_TRY.value)):
    #   run_clustering(k, _FIXED_EPS.value, datapoints)

    print(f'\n# Evaluation with k = {_FIXED_K.value} and '
          f'varying epsilon in {list(map(float, _EPS_TO_TRY.value))}')
    print(eval_head)
    for epsilon in list(map(float, _EPS_TO_TRY.value)):
      run_clustering(_FIXED_K.value, epsilon, datapoints)

if __name__ == "__main__":
    app.run(main)