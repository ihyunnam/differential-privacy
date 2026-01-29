# Input: LMSYS 25k csv file from Su & Aarzoo
# Output: clustering_params.Data format

# from typing import Sequence
from absl import app
from absl import flags
import numpy as np
import json
import math
import matplotlib.pyplot as plt

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
# _EPS_TO_TRY = flags.DEFINE_list(
#     'epsilon_to_try', '0.1, 0.5, 1.0, 1.5, 2.0',
#     'List of epsilon values to use when experimenting with varying epsilon.')
_MAX_DEPTH = flags.DEFINE_list(
    'max_depth', '2,4,6,8,10,12,14,16,18,20',
    'Value of max_depth when experimenting with varying epsilon.')
_MIN_NUM_POINTS_IN_BRANCHING_NODE = flags.DEFINE_list(
    'min_num_points_in_branching_node', '20, 40, 60, 80, 100, 200, 300',
    'Value of min_num_points_in_branching_node when experimenting with varying epsilon.')
_MIN_NUM_POINTS_IN_NODE = flags.DEFINE_list(
    'min_num_points_in_node', '20, 40, 60, 80, 100, 200, 300',
    'Value of min_num_points_in_node when experimenting with varying epsilon.')

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

    data = clustering_params.Data(datapoints, radius)  # This returns type Data
    # data_clipped = data.clip_by_radius()  # This returns type Points not Data
    
    # SANITY CHECK: max ~= radius 
    norms = np.linalg.norm(data.datapoints, axis=1)
    print(f"norms min: {norms.min()}, norms max: {norms.max()}")

    return clustering_params.Data(data.datapoints, radius)

eval_head = ('| epsilon | max_depth | actual_depth | min_branch | min_node | cluster_frac | loss |')

def run_clustering(min_num_points_in_branching_node: int, min_num_points_in_node: int, max_depth: int, k: int, eps: float, data: clustering_params.Data) -> tuple[float, float, int, int]:
  privacy_param = clustering_params.DifferentialPrivacyParam(
      epsilon=eps, delta=1e-6)
  tree_param = clustering_params.TreeParam(
      min_num_points_in_branching_node=min_num_points_in_branching_node,
      min_num_points_in_node=min_num_points_in_node,
      max_depth=max_depth)

  # IHYUN: We want to aim for coreset_weight_avg ~ 150 like in URANIA
  # IHYUN: min_branch, min_node had marginal impact on coreset_weight_avg.
  # TODO try larger max_depths so there's more possible clusters?
  cluster_occupancy, actual_depth, coreset_weight_avg, clustering_result = (
      clustering_algorithm.private_lsh_clustering(
          k,
          data,
          privacy_param,
          tree_param=tree_param))

  norm_loss = clustering_result.loss / data.num_points
  return cluster_occupancy, coreset_weight_avg, actual_depth, norm_loss

def sanity_test_letter_recog() -> None:
    from ucimlrepo import fetch_ucirepo 
  
    # fetch dataset 
    letter_recognition = fetch_ucirepo(id=59) 
    
    # data (as pandas dataframes) 
    X = letter_recognition.data.features 
    y = letter_recognition.data.targets 

def main(argv):
    del argv  # unused
    datapoints = prep_data()

    print(eval_head)

    records = []

    print(f"fixed k: {_FIXED_K.value}")

    def on_pick(event):
        ind = event.ind[0]  # index of picked point
        r = records[ind]
        print(
            f"\nPicked point #{ind}\n"
            # f"  epsilon      = {r['epsilon']}\n"
            f"  max_depth    = {r['max_depth']}\n"
            f"  actual_depth    = {r['actual_depth']}\n"
            f"  min_branch   = {r['min_branch']}\n"
            f"  min_node     = {r['min_node']}\n"
            f"  cluster_frac = {r['cluster_frac']:.3f}\n"
            f"  loss         = {r['loss']:.4f}\n"
        )

    # for epsilon in map(float, _FIXED_EPS.value):
    for max_depth in map(int, _MAX_DEPTH.value):
        for min_branch in map(int, _MIN_NUM_POINTS_IN_BRANCHING_NODE.value):
            for min_node in map(int, _MIN_NUM_POINTS_IN_NODE.value):
                if min_node > min_branch:
                    continue

                cluster_occupancy, coreset_weight_avg, actual_depth, norm_loss = run_clustering(
                    min_branch,
                    min_node,
                    max_depth,
                    _FIXED_K.value,
                    _FIXED_EPS.value,
                    datapoints,
                )

                records.append({
                    # "epsilon": epsilon,
                    "max_depth": max_depth,
                    "actual_depth": actual_depth,
                    "min_branch": min_branch,
                    "min_node": min_node,
                    "coreset_weight_avg": coreset_weight_avg,
                    "cluster_frac": cluster_occupancy,  # ∈ [0,1]
                    "loss": norm_loss,                   # normalized loss
                })

    cluster_frac = np.array([r["cluster_frac"] for r in records])
    coreset_weight_avg = np.array([r["coreset_weight_avg"] for r in records])
    loss = np.array([r["loss"] for r in records])
    # epsilons = np.array([r["epsilon"] for r in records])
    max_depths = np.array([r["max_depth"] for r in records])
    actual_depths = np.array([r["actual_depth"] for r in records])
    min_branch = np.array([r["min_branch"] for r in records])

    # Sort records by cluster occupancy only (high → low)
    records_sorted = sorted(
        records,
        key=lambda r: r["cluster_frac"],
        reverse=True
    )

    for r in records_sorted:
        print(
            f"occupancy={100 * r['cluster_frac']:.2f}% | "
            f"coreset_weight_avg={r['coreset_weight_avg']} | "
            # f"ε={r['epsilon']} | "
            f"max_depth={r['max_depth']} | "
            f"actual_depth={r['actual_depth']} | "
            f"min_branch={r['min_branch']} | "
            f"min_node={r['min_node']} | "
            f"loss={r['loss']:.4f}"
        )
    plt.figure(figsize=(7, 5))

    plt.xlabel("Fraction of clusters occupied")
    plt.ylabel("Normalized k-means loss")
    plt.title("Loss vs Cluster Occupancy (Pareto View)")

    scatter = plt.scatter(
        cluster_frac,
        loss,
        # c=np.log10(epsilons),
        s=30 + 10 * max_depths,
        picker=True,
    )

    plt.gcf().canvas.mpl_connect("pick_event", on_pick)

    # cbar = plt.colorbar(scatter)
    # cbar.set_label("log10(epsilon)")

    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    app.run(main)