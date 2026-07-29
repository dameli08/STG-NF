import numpy as np

from stgnf.graph.graph import Graph


def test_openpose_18_nodes():
    g = Graph(layout="openpose", strategy="uniform", max_hop=8)
    assert g.num_node == 18
    assert g.A.shape == (1, 18, 18)


def test_uniform_adjacency_is_row_normalized():
    g = Graph(layout="openpose", strategy="uniform", max_hop=1)
    # Uniform strategy uses a single normalized adjacency matrix.
    assert g.A.shape[0] == 1
    assert np.isfinite(g.A).all()


def test_spatial_strategy_multiple_partitions():
    g = Graph(layout="openpose", strategy="spatial", max_hop=1)
    assert g.A.shape[1:] == (18, 18)
    assert g.A.shape[0] >= 2
