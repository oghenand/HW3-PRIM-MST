import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """

        V = set(range(len(self.adj_mat))) # set of vertices
        S = set() # initialize set of nodes in MST
        T = set() # initialize set of edges in MST

        s = next(iter(V)) # initialize from a given node s
        pq = [] # empty priority queue
        pi = {v: float('inf') for v in V} # initialize dict for cost of cheapest edge
        pi[s] = 0 # cost of starting node is zero (tree starts with it)
        pred = {v: None for v in V} # which node in the tree cheapest edge comes from (null in the beginning)
        # the above is stored as an edge for easy addition to T (later in the code)
        # add nodes to priority queue
        for v in V:
            heapq.heappush(pq, (pi[v], v)) # add node and cost
    
        # while loop to create mst
        while pq:
            cost, u = heapq.heappop(pq) # get lowest cost node

            if u in S: # if u already in S/processed, continue
                continue

            S.add(u) # add to S if not processed
            if pred[u] is not None: # add edge to T if pred has cheapest edge for u
                T.add(pred[u])
            
            # check for neighbors of u
            for v in V:
                if v not in S and self.adj_mat[u,v] != 0:
                    # find edge cost between u and neighbor v
                    edge_cost = self.adj_mat[u,v]
                    # if edge cost lower than what's currently stored in pi[v], update
                    if edge_cost < pi[v]:
                        # update edge cost
                        pi[v] = edge_cost
                        # update cheapest parent edge
                        pred[v]  = (u,v)
                        heapq.heappush(pq, (edge_cost, v)) # update priority queue

        # initialize mst as array of zeros
        self.mst = np.zeros((len(V), len(V)))
        # add mst edges
        for mst_edge in T:
            self.mst[mst_edge] = self.adj_mat[mst_edge]
            # add reverse since symmetric for undirected graph
            self.mst[tuple(reversed(mst_edge))] = self.adj_mat[tuple(reversed(mst_edge))]
