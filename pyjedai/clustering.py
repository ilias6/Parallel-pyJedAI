from queue import PriorityQueue
from networkx import connected_components, Graph
from time import time
from tqdm.autonotebook import tqdm

class ConnectedComponentsClustering:
    """Creates the connected components of the graph. \
        Applied to graph created from entity matching. \
        Input graph consists of the entity ids (nodes) and the similarity scores (edges).
    """

    _method_name: str = "Connected Components Clustering"
    _method_short_name: str = "CCC"
    _method_info: str = "Gets equivalence clusters from the " + \
                    "transitive closure of the similarity graph."

    def __init__(self) -> None:
        self.execution_time: float

    def process(self, graph: Graph) -> list:
        """NetworkX Connected Components Algorithm in the produced graph.

        Args:
            graph (Graph): Consists of the entity ids (nodes) and the similarity scores (edges).

        Returns:
            list: list of clusters
        """
        start_time = time()
        clusters = list(connected_components(graph))
        self.execution_time = time() - start_time
        return clusters

    def method_configuration(self) -> dict:
        """Returns configuration details
        """
        return {
            "name" : self._method_name,
            "parameters" : self._configuration(),
            "runtime": self.execution_time
        }

    def _configuration(self) -> dict:
        return {}

    def report(self) -> None:
        """Prints Block Building method configuration
        """
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            "\nRuntime: {:2.4f} seconds".format(self.execution_time)
        )


class UniqueMappingClustering:
    """Prunes all edges with a weight lower than t, sorts the remaining ones in
        decreasing weight/similarity and iteratively forms a partition for
        the top-weighted pair as long as none of its entities has already
        been matched to some other.
    """

    _method_name: str = "Unique Mapping Clustering"
    _method_short_name: str = "UMC"
    _method_info: str = "Prunes all edges with a weight lower than t, sorts the remaining ones in" + \
                        "decreasing weight/similarity and iteratively forms a partition for" + \
                        "the top-weighted pair as long as none of its entities has already" + \
                        "been matched to some other."

    def __init__(self, similarity_threshold: float = 0.1) -> None:
        """Unique Mapping Clustering Constructor

        Args:
            similarity_threshold (float, optional): Prunes all edges with a weight lower than this. Defaults to 0.1.
        """
        self.similarity_threshold: float = similarity_threshold
        self.execution_time: float

    def process(self, graph: Graph) -> list:
        """NetworkX Connected Components Algorithm in the produced graph.

        Args:
            graph (Graph): Consists of the entity ids (nodes) and the similarity scores (edges).

        Returns:
            list: list of clusters
        """
        start_time = time()
        matched_entities = {}
        new_graph = Graph()
        priority_queue = PriorityQueue(maxsize=graph.number_of_edges()*2)
        for x in graph.edges(data=True):
            if x[2]['weight'] > self.similarity_threshold:
                priority_queue.put_nowait((x[2]['weight'], x[0], x[1]))

        while not priority_queue.empty():
            sim, entity_1, entity_2 = priority_queue.get()
            if entity_1 in matched_entities or entity_2 in matched_entities:
                continue
            new_graph.add_edge(entity_1, entity_2, weight=sim)
            matched_entities |= {entity_1, entity_2}

        clusters = ConnectedComponentsClustering().process(new_graph)
        self.execution_time = time() - start_time
        return clusters

    def method_configuration(self) -> dict:
        """Returns configuration details
        """
        return {
            "name" : self._method_name,
            "parameters" : self._configuration(),
            "runtime": self.execution_time
        }

    def _configuration(self) -> dict:
        return {}

    def report(self) -> None:
        """Prints Block Building method configuration
        """
        print(
            "Method name: " + self._method_name +
            "\nMethod info: " + self._method_info +
            "\nRuntime: {:2.4f} seconds".format(self.execution_time)
        )
