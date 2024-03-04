from enum import Enum
import numpy as np


class LinkageType(Enum):
    """An enumeration of the different types of linkage methods."""

    SINGLE = "single"
    COMPLETE = "complete"
    AVERAGE = "average"
    CENTROID = "centroid"
    WARD = "ward"


class LinkageCalculator:
    def __init__(self, linkage_type: LinkageType) -> None:
        self.linkage_type = linkage_type

    def single_linkage(self, cluster_a: list[np.ndarray], cluster_b: list[np.ndarray]) -> float:
        """Calculates the single linkage between two clusters."""
        raise NotImplementedError("TODO(@larkin): Implement single linkage")

    def complete_linkage(self, cluster_a: list[np.ndarray], cluster_b: list[np.ndarray]) -> float:
        """Calculates the complete linkage between two clusters."""
        raise NotImplementedError("TODO(@larkin): Implement complete linkage")

    def average_linkage(self, cluster_a: list[np.ndarray], cluster_b: list[np.ndarray]) -> float:
        """Calculates the average linkage between two clusters."""
        raise NotImplementedError("TODO(@larkin): Implement average linkage")

    def centroid_linkage(self, cluster_a: list[np.ndarray], cluster_b: list[np.ndarray]) -> float:
        """Calculates the centroid linkage between two clusters."""
        raise NotImplementedError("TODO(@larkin): Implement centroid linkage")

    def ward_linkage(self, cluster_a: list[np.ndarray], cluster_b: list[np.ndarray]) -> float:
        """Calculates the ward linkage between two clusters."""
        raise NotImplementedError("TODO(@larkin): Implement centroid linkage")

    def calculate(
        cluster_a,
        cluster_b,
        other_cluster,
        linkage_type,
        data,
        distance_matrix,
        index_to_cluster_id,
    ):
        # Identify the points belonging to each cluster
        points_in_a = [
            index for index, cluster_id in index_to_cluster_id.items() if cluster_id == cluster_a
        ]
        points_in_b = [
            index for index, cluster_id in index_to_cluster_id.items() if cluster_id == cluster_b
        ]
        points_in_other = [
            index
            for index, cluster_id in index_to_cluster_id.items()
            if cluster_id == other_cluster
        ]

        if linkage_type == LinkageType.SINGLE:
            # Find the minimum distance between points in the merged cluster (a+b) and the other cluster
            distances = [
                distance_matrix[a][o] for a in points_in_a + points_in_b for o in points_in_other
            ]
            return min(distances) if distances else float("inf")

        elif linkage_type == LinkageType.COMPLETE:
            # Find the maximum distance between points in the merged cluster (a+b) and the other cluster
            distances = [
                distance_matrix[a][o] for a in points_in_a + points_in_b for o in points_in_other
            ]
            return max(distances) if distances else float("inf")

        elif linkage_type == LinkageType.AVERAGE:
            # Calculate the average distance between all points in the merged cluster (a+b) and the other cluster
            distances = [
                distance_matrix[a][o] for a in points_in_a + points_in_b for o in points_in_other
            ]
            return np.mean(distances) if distances else float("inf")
