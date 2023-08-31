"""
@Author: Simona Bernardi
@Date: updated 24/02/2021

Graph comparison module
"""

import numpy as np
from scipy.stats import entropy #it is used to compute the KLD measure
from scipy.spatial import distance #it is used to compute the JS measure

#from src.graph_discovery import Graph

DEBUG = True


class GraphComparator:
    """Operator that compares two graphs"""

    def __init__(self,gr1,gr2):
        # First operand
        self.graph1 = gr1
        # Second operand
        self.graph2 = gr2

    def sortGraph(self, graph):
        # insertion sort
        nodes = graph.nodes
        nodesFreq = graph.nodesFreq
        edges = graph.matrix

        for colidx in range(1, len(nodes)):

            pivot = []
            pivot.append(nodes[colidx])
            pivot.append(nodesFreq[colidx])
            pivot.append(edges[colidx])
            # print("Pivot:",pivot)
            position = colidx

            # compare node labels
            while position > 0 and nodes[position - 1] > pivot[0]:
                # move row one step down
                nodes[position] = nodes[position - 1]
                edges[position] = edges[position - 1]

                position = position - 1

            # put the pivot in the found position
            nodes[position] = pivot[0]
            nodesFreq[position] = pivot[1]
            edges[position] = pivot[2]

            for i in range(len(edges)):
                swapel = edges[i][colidx]
                edges[i][colidx] = edges[i][position]
                edges[i][position] = swapel

    def expandGraph(self, graph, position, vertex):
        # Different from zero to differentiate from the absence of arc,
        # but presence of the node
        wildcard = '-1'
        # Insert the new vertex in the list of nodes
        graph.nodes = np.insert(graph.nodes, position, vertex)
        graph.nodesFreq = np.insert(graph.nodesFreq, position, wildcard)
        # Insert the new column in the matrix
        graph.matrix = np.insert(graph.matrix, position, wildcard, axis=1)
        # Insert the new row in the matrix
        graph.matrix = np.insert(graph.matrix, position, wildcard, axis=0)

    def normalizeGraphs(self):
        first = self.graph1
        second = self.graph2

        # Sort the graphs according to the node name list: maybe sorting is not needed
        self.sortGraph(first)
        self.sortGraph(second)

        # Union of the nodes
        nodesU = np.union1d(first.nodes, second.nodes)

        # Compare the node list and possibly extend the model(s)
        for i in range(nodesU.size):
            if (first.nodes.size > i) and (first.nodes[i] != nodesU[i]) or (
                    first.nodes.size <= i):
                self.expandGraph(first, i, nodesU[i])
            if (second.nodes.size > i) and (second.nodes[i] != nodesU[i]) or (
                    second.nodes.size <= i):
                self.expandGraph(second, i, nodesU[i])

    def compareGraphs(self):  # signature only because it is overriden
        return 0



# Strategy pattern (variant)
class GraphHammingDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic 1: Structural-based distance
        The two matrix arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()
        # Just the adjacency matrix
        dim = min(len(first), len(second))
        # Setting a counter vector to zero
        counter = np.zeros(dim)
        # Count if both elements are either positive or zero
        counter = np.where(((first > 0) & (second > 0)) |
                           ((first == 0) & (second == 0)), counter + 1, counter)

        distance = 1.0 - np.sum(counter) / float(dim)

        return distance  #returns the dissimilarity (distance)


class GraphCosineDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic 2: Frequency-based similarity (cosine)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """

        # Convert into arrays the node frequencies and matrices
        first = np.concatenate(
            (self.graph1.nodesFreq, self.graph1.matrix.flatten()), axis=None)
        second = np.concatenate(
            (self.graph2.nodesFreq, self.graph2.matrix.flatten()), axis=None)
        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)
        # Normalization factor
        nfactor = 1.0
        sp = first * second / nfactor
        # Frobenius norm (L2-norm Euclidean)
        norm1 = np.linalg.norm(first)
        norm2 = np.linalg.norm(second)
        # Compute the product
        cosinus = np.sum(sp) / (norm1 * norm2)

        return 1 - cosinus #returns the dissimilarity (distance)

class GraphKLDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic 3: Kullback-Leibler divergence (of the matrices)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """
        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()

        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (probability distributions are needed)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the KLD of first  w.r.t second 
        kld = entropy(first,second,base=2)

        return kld  #returns the dissimilarity (distance)

class GraphJSDissimilarity(GraphComparator):

    def compareGraphs(self):
        """
        Heuristic 4: Jensen-Shannon divergence (of the matrices)
        The two arrays are assumed
        --  of the same length (minimum of the length is chosen)
        --  corresponding to the same ordered set of nodes:
            graph normalization has to be done before!
        """
        # Get just the matrices and convert into arrays
        first = self.graph1.matrix.flatten()
        second = self.graph2.matrix.flatten()

        # Set -1 entries to zero
        first = np.where((first < 0), first * 0, first)
        second = np.where((second < 0), second * 0, second)

        # Normalizes the matrices (probability distributions are needed)
        first = first / (first.sum())
        second = second / (second.sum())

        # Compute the KLD of first  w.r.t second 
        jsd = distance.jensenshannon(first,second,base=2)

        return jsd  #returns the dissimilarity (distance)