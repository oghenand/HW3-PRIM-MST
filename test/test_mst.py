import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances
import heapq


def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    edge_count = 0
    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
            if mst[i,j] != 0:
                edge_count += 1
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    # test case for edges in a mst
    num_nodes = adj_mat.shape[0]
    num_mst_edges_true = num_nodes - 1
    assert num_mst_edges_true == edge_count, "MST does not have n-1 edges! incorrect!"

    def is_connected(adj_mat, mst):
        # Function to check if mst is connected (it should be!)
        # done with dfs
        num_nodes = len(mst)
        if num_nodes == 0:
            return True # empty graph is connected

        visited = set()
        def dfs(node): # function to run dfs from a given node
            visited.add(node) # add node to visited
            for neighbor in range(num_nodes): # visit nodes that are neighboring, have non-neg. weights and not in visited
                if mst[node][neighbor] != 0 and neighbor not in visited:
                    # recursively run dfs on neighbors -- stops when all possible nodes that can be visted are visited
                    dfs(neighbor)
        # run dfs from random start point
        dfs(0)
        # True if visited is equal to number of nodes, meaning MST is connected
        return len(visited) == len(adj_mat)
    
    assert is_connected(adj_mat, mst), "MST is not connected, incorrect!"

    # test case to check if minimum edge is in MST (it should be!)
    non_zero_weights = adj_mat[adj_mat!=0] # find all non-zero weights
    # if the graph actually has non-zero weights and more than one node, then do check 
    # if one node in graph, mst should have no weights -- this handles that here.
    if len(non_zero_weights) > 0 and adj_mat.shape[0] > 1:
        min_weight = non_zero_weights.min()
        assert min_weight in mst, "Minumum weight not in MST, incorrect!"

def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695)


def test_mst_student():
    """
    
    TODO: Write at least one unit test for MST construction.
    
    """
    # example graph and adj mat
    my_adj_mat = np.array([
        [0, 5, 1, 2],
        [5, 0, 4, 0],
        [1, 4, 0, 3],
        [2, 0, 3, 0]
    ])

    # create graph object and construct mst
    g = Graph(my_adj_mat)
    g.construct_mst()

    # example ground truth mst -- expected mst weight of 7
    my_mst = np.array([
        [0, 0, 1, 2],
        [0, 0, 4, 0],
        [1, 4, 0, 0],
        [2, 0, 0, 0],
    ])
    # run test cases
    check_mst(g.adj_mat, g.mst, my_mst.sum()/2)

def test_mst_single_node():
    # checking handling of single node graph (mst is single node with no edges)
    my_adj_mat = np.array([
        [7]
    ])
    g = Graph(my_adj_mat)
    g.construct_mst()
    # check if correct mst is found (node itself with no edges)
    check_mst(g.adj_mat, g.mst, expected_weight=0)