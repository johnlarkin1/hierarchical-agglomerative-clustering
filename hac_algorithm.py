import re
import numpy as np
import pandas as pd
import time
from enum import Enum
from typing import Callable, NamedTuple
from linkage import LinkageCalculator, LinkageType
from Levenshtein import distance as levenshtein_distance
from typing import List
from linkage import LinkageType
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
import seaborn as sns


DATA_FILE_PATH_EASY = "data/input/data_easy.csv"
DATA_FILE_PATH_CPX = "data/input/data_complex.csv"
SHOULD_USE_THIRD_PARTY = True
SHOULD_STOP_EARLY = False


class DistanceType(Enum):
    """An enumeration fo the different types of distance methods we can compute."""

    LEVENSHTEIN = "levenshtein"


class HACAlgoInput(NamedTuple):
    data_file_path: str
    should_use_third_party: bool
    should_stop_early: bool
    linkage_type: LinkageType


def timing_decorator(func):
    """
    A decorator that logs the execution time of the function it decorates.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Call the decorated function
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.4f} seconds to execute.")
        return result

    return wrapper


def compute_distance(row1: pd.Series, row2: pd.Series) -> int:
    """
    Computes the distance between a given row and another row.

    This method helps to abstract out how we're acutally computing distance,
    in case of future iterations / improvements.
    """

    distance = levenshtein_distance(row1["retailer_nm_modified"], row2["retailer_nm_modified"])
    return distance


def calculate_new_distance(
    cluster_a: int,
    cluster_b: int,
    other_cluster: int,
    linkage_type: LinkageType,
    distance_matrix: np.ndarray,
    index_to_cluster_id: dict[int, int],
) -> float:
    # Identify the points belonging to each cluster
    points_in_a = [
        index for index, cluster_id in index_to_cluster_id.items() if cluster_id == cluster_a
    ]
    points_in_b = [
        index for index, cluster_id in index_to_cluster_id.items() if cluster_id == cluster_b
    ]
    points_in_other = [
        index for index, cluster_id in index_to_cluster_id.items() if cluster_id == other_cluster
    ]
    distances = [distance_matrix[a][o] for a in points_in_a + points_in_b for o in points_in_other]

    if linkage_type == LinkageType.SINGLE:
        # Find the minimum distance between points in the merged cluster (a+b) and the other cluster
        return min(distances) if distances else float("inf")

    elif linkage_type == LinkageType.COMPLETE:
        # Find the maximum distance between points in the merged cluster (a+b) and the other cluster
        return max(distances) if distances else float("inf")

    elif linkage_type == LinkageType.AVERAGE:
        # Calculate the average distance between all points in the merged cluster (a+b) and the other cluster
        return np.mean(distances) if distances else float("inf")
    else:
        raise NotImplementedError(f"Linkage type {linkage_type} not implemented")


class HierarchicalAgglomerativeClustering:
    """A from scratch implementation of Hierarchical Agglomerative Clustering algorithm."""

    data_file_path: str
    linkage_type: LinkageType
    raw_data: pd.DataFrame
    processed_data: pd.DataFrame
    steps: list[Callable[[], pd.DataFrame]]

    should_enforce_stopping_criteria: bool
    should_use_third_party: bool

    output_txt_path: str
    output_dendrogram_path: str

    def __init__(
        self,
        data_file_path: str,
        linkage_type: LinkageType = LinkageType.SINGLE,
        should_enforce_stopping_criteria: bool = False,
        should_use_third_party: bool = False,
    ) -> None:
        """Initializes our HAC algorithm with the necessary parameters."""
        self.data_file_path = data_file_path
        self.linkage_type = linkage_type
        self.should_enforce_stopping_criteria = should_enforce_stopping_criteria
        self.should_use_third_party = should_use_third_party
        self.linkage_calculator = LinkageCalculator(linkage_type)
        self.raw_data = pd.read_csv(data_file_path)
        file_type = "complex" if "complex" in data_file_path.lower() else "simple"
        dist_algo = "leven"
        impl_type = "3rd_party" if should_use_third_party else "scratch"
        core_name = f"{impl_type}_{file_type}_{linkage_type.value}_{dist_algo}"
        self.output_txt_path = f"data/output/{core_name}.txt"
        self.output_dendrogram_path = f"data/output/dendrogram_{core_name}.png"
        self.steps = [
            self.preprocess,
            self.generate_ground_truth,
            self.hierarchical_clustering,
            self.generate_cluster_id_mapping,
            self.compute_scores,
        ]

    def preprocess(self) -> pd.DataFrame:
        """
        Preprocesses our data.

        This is going to do very simple cleansing on our retailer_nm_modified data
        by removing any special characters and converting it to lowercase.
        Note, we're operating under the principal that we don't know how the data was originally
        tainted.
        """
        # target column that we're focused on cleaning up
        tgt_col = "retailer_nm_modified"

        self.processed_data = self.raw_data.copy()

        # Fill NA values with empty string
        self.processed_data["retailer_nm_modified"] = self.processed_data[
            "retailer_nm_modified"
        ].fillna("")

        # Remove spaces and convert to lowercase
        self.processed_data[tgt_col] = self.processed_data[tgt_col].apply(
            lambda x: str(x).replace(" ", "").lower()
        )
        # Remove numbers from names
        self.processed_data[tgt_col] = self.processed_data[tgt_col].apply(
            lambda x: re.sub(r"\d+", "", str(x))
        )
        return self.processed_data

    def generate_ground_truth(self) -> pd.DataFrame:
        """
        Generates the ground truth for our data.

        We'll generate a unique id for each retailer_nm.
        """

        unique_retailers = self.processed_data["retailer_nm"].unique()
        retailer_to_label = {retailer: label for label, retailer in enumerate(unique_retailers)}
        # Add a new column to our data that maps the retailer_nm to a unique label
        self.processed_data["ground_truth_label"] = self.processed_data["retailer_nm"].map(
            retailer_to_label
        )
        return self.processed_data

    @timing_decorator
    def hierarchical_clustering(
        self,
        should_enforce_stopping_criteria: bool = False,
    ) -> pd.DataFrame:
        if self.should_use_third_party:
            return self.hierarchical_clustering_from_third_party()
        return self.hierarchical_clustering_from_scratch(should_enforce_stopping_criteria)

    def hierarchical_clustering_from_third_party(self) -> pd.DataFrame:
        """
        Performs the hierarchical clustering algorithm using a third-party library.
        """
        retailer_nm_modified = self.processed_data["retailer_nm_modified"].values

        # Calculate the Levenshtein distance matrix in a condensed form
        n = len(retailer_nm_modified)
        condensed_dist_matrix = np.zeros(n * (n - 1) // 2)
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                condensed_dist_matrix[k] = levenshtein_distance(
                    retailer_nm_modified[i], retailer_nm_modified[j]
                )
                k += 1

        Z = linkage(condensed_dist_matrix, self.linkage_type.value)

        desired_clusters = len(self.processed_data["retailer_nm"].unique())
        cluster_labels = fcluster(Z, desired_clusters, criterion="maxclust")
        self.processed_data["cluster_id"] = cluster_labels

        sns.set_style("darkgrid")
        plt.figure(figsize=(10, 7))
        dendrogram(Z, labels=self.processed_data["retailer_nm_modified"].values)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Retailer Identifier")
        plt.ylabel("Distance")
        plt.savefig(self.output_dendrogram_path)
        plt.show()

    def hierarchical_clustering_from_scratch(
        self,
        should_enforce_stopping_criteria: bool = False,
    ) -> pd.DataFrame:
        """
        Performs the hierarchical clustering algorithm.

        1. Builds our distance matrix.
        2. Initiative all points as their own cluster.
        3. Finds the closest clusters.
        4. Iteratively merges the closest clusters.
        5. Updates the distance matrix.
        6. Repeat steps 3-5 until we have a single cluster.
        """

        # 1. Build distance matrix
        @timing_decorator
        def build_distance_matrix(data: pd.DataFrame) -> np.ndarray:
            """
            Builds the distance matrix for our data.

            The distance matrix is a square matrix that contains the pairwise distances
            between each point in our data.
            """

            num_rows = len(self.processed_data["retailer_nm_modified"].values)
            distance_matrix = np.zeros((num_rows, num_rows))
            for i in range(num_rows):
                for j in range(i + 1, num_rows):
                    distance = compute_distance(data.iloc[i], data.iloc[j])
                    # distance matrix is symmetrical
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
            return distance_matrix

        distance_matrix = build_distance_matrix(self.processed_data)
        n = len(self.processed_data)
        cluster_id_to_dendrogram_index = {i: i for i in range(n)}
        next_cluster_index = n
        dendrogram_data = []

        # 2. Initialize all points as their own cluster
        self.processed_data["cluster_id"] = self.processed_data.index
        index_to_cluster_id: dict[int, int] = {
            i: cluster_id for i, cluster_id in enumerate(self.processed_data["cluster_id"])
        }

        @timing_decorator
        def find_closest_clusters(
            distance_matrix: np.ndarray,
            index_to_cluster_id: dict[int, int],
        ) -> tuple[int, int, int, int]:
            """
            Finds the two closest clusters in our data.

            We'll use the distance matrix to find the two closest clusters.
            """
            min_val = np.inf
            cluster_index_a, cluster_index_b = -1, -1
            for i in range(distance_matrix.shape[0]):
                for j in range(i + 1, distance_matrix.shape[1]):  # Ensure i != j
                    # Check if i and j belong to different clusters before comparing distances
                    if (
                        distance_matrix[i, j] < min_val
                        and index_to_cluster_id[i] != index_to_cluster_id[j]
                    ):
                        min_val = distance_matrix[i, j]
                        cluster_index_a, cluster_index_b = i, j

            cluster_a_id = index_to_cluster_id[cluster_index_a]
            cluster_b_id = index_to_cluster_id[cluster_index_b]
            # Additional check to ensure cluster IDs are distinct could be added here, if necessary
            return cluster_a_id, cluster_b_id, cluster_index_a, cluster_index_b

        @timing_decorator
        def merge_closest_clusters(
            cluster_a: int,
            cluster_b: int,
            cluster_index_a: int,
            cluster_index_b: int,
        ) -> pd.DataFrame:
            """
            Merges the two closest clusters in our actual dataframe.
            We don't touch our distance matrix yet.

            We'll merge the two closest clusters and update the cluster_id column
            in our data.
            """
            nonlocal next_cluster_index

            # Update the cluster_id for all points in cluster_b
            self.processed_data.loc[
                self.processed_data["cluster_id"] == cluster_b, "cluster_id"
            ] = cluster_a
            merge_distance = distance_matrix[cluster_index_a, cluster_index_b]
            new_cluster_size = len(
                self.processed_data[self.processed_data["cluster_id"] == cluster_a]
            )
            dendrogram_data.append(
                [
                    cluster_id_to_dendrogram_index[cluster_a],
                    cluster_id_to_dendrogram_index[cluster_b],
                    merge_distance,
                    new_cluster_size,
                ]
            )

            cluster_id_to_dendrogram_index[cluster_a] = next_cluster_index
            cluster_id_to_dendrogram_index[cluster_b] = next_cluster_index

            next_cluster_index += 1  # Prepare for the next merge
            return self.processed_data

        @timing_decorator
        def update_distance_matrix(dist_matrix: np.ndarray, cluster_a: int, cluster_b: int) -> None:
            # We always merge cluster_b into cluster_a
            for idx, cluster_id in list(index_to_cluster_id.items()):
                if cluster_id == cluster_b:
                    index_to_cluster_id[idx] = cluster_a

            # Set diagonal to np.inf to ignore self-distances
            np.fill_diagonal(dist_matrix, np.inf)

            # Recompute distances for the new cluster
            # We only need to update the distances for when
            # the distance matrix is referencing cluster_a or cluster_b
            for i in range(len(dist_matrix)):
                for j in range(len(dist_matrix)):
                    # Get the cluster IDs for points i and j
                    cluster_id_i = index_to_cluster_id.get(i)
                    cluster_id_j = index_to_cluster_id.get(j)

                    # If i or j is part of the newly merged cluster, recalculate the distance
                    if cluster_id_i == cluster_a or cluster_id_j == cluster_a:
                        new_distance = calculate_new_distance(
                            cluster_a,
                            cluster_b,
                            cluster_id_j if cluster_id_i == cluster_a else cluster_id_i,
                            self.linkage_type,
                            dist_matrix,
                            index_to_cluster_id,
                        )
                        dist_matrix[i][j] = new_distance
                        dist_matrix[j][i] = new_distance
            return dist_matrix, index_to_cluster_id

        # Now we loop until we have a single cluster
        unique_retailer_count = len(self.processed_data["retailer_nm"].unique())
        while len(self.processed_data["cluster_id"].unique()) > (
            unique_retailer_count
            if (should_enforce_stopping_criteria or self.should_enforce_stopping_criteria)
            else 1
        ):
            print("Number of clusters:", len(self.processed_data["cluster_id"].unique()))
            start_time = time.time()

            # 3. Find the closest clusters
            cluster_a, cluster_b, cluster_index_a, cluster_index_b = find_closest_clusters(
                distance_matrix, index_to_cluster_id
            )
            # 4. Merge the closest clusters in our
            self.processed_data = merge_closest_clusters(
                cluster_a, cluster_b, cluster_index_a, cluster_index_b
            )
            # 5. Update the distance matrix
            distance_matrix, index_to_cluster_id = update_distance_matrix(
                distance_matrix, cluster_a, cluster_b
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Elapsed time of clustering loop:", elapsed_time, "seconds")

        # Only try to show the dendrogram if we have the full merge history
        if not should_enforce_stopping_criteria and not self.should_enforce_stopping_criteria:
            sns.set_style("darkgrid")
            plt.figure(figsize=(10, 7))
            linkage_matrix = np.array(dendrogram_data)
            np.savetxt("data/output/linkage_matrix.csv", linkage_matrix, delimiter=",")
            print("Dendrogram Data:", dendrogram_data)

            n_clusters = linkage_matrix.shape[0] + 1
            labels = [f"Sample {i+1}" for i in range(n_clusters)]

            dendrogram(
                linkage_matrix,
                orientation="top",
                labels=labels,
                distance_sort="descending",
                show_leaf_counts=True,
                leaf_rotation=90.0,
                leaf_font_size=8.0,
                color_threshold=0.7 * max(linkage_matrix[:, 2]),
                above_threshold_color="grey",
            )

            plt.title("Hierarchical Clustering Dendrogram", fontsize=16)
            plt.xlabel("Index", fontsize=10)
            plt.ylabel("Distance", fontsize=14)
            plt.savefig(self.output_dendrogram_path)
            plt.show()

        return self.processed_data

    def generate_cluster_id_mapping(self) -> pd.DataFrame:
        """
        Generates a mapping of cluster IDs to ground truth labels.

        We do this by grouping our data by cluster_id and ground_truth_label,
        and then counting the occurrences. We then sort the data to ensure the
        mode (highest count) comes first, and then drop duplicates to ensure
        a one-to-one mapping.
        """
        # Group by cluster_id and ground_truth_label, and count occurrences
        grouped = (
            self.processed_data.groupby(["cluster_id", "ground_truth_label"])
            .size()
            .reset_index(name="count")
        )

        # Sort the grouped data to ensure the mode (highest count) comes first
        grouped_sorted = grouped.sort_values(by=["cluster_id", "count"], ascending=[True, False])

        # Drop duplicates to ensure one-to-one mapping, keeping the first occurrence (the mode)
        unique_mapping = grouped_sorted.drop_duplicates(subset="cluster_id", keep="first")

        mapping_dict = pd.Series(
            unique_mapping.ground_truth_label.values, index=unique_mapping.cluster_id
        ).to_dict()

        print("mapping_dict", mapping_dict)

        self.processed_data["orig_cluster_id"] = self.processed_data["cluster_id"]
        self.processed_data["cluster_id"] = self.processed_data["cluster_id"].map(mapping_dict)
        return self.processed_data

    def compute_scores(self) -> None:
        """
        This is really part 4 of our prompt.

        > Measure the accuracy of your solution.
        > Generate a confusion matrix to calculate the precision
        > and recall for each predicted “class”.
        > Use the ground_truth_label that was generated in an earlier step.

        ## Precision
        Precision measures the accuracy of positive predictions.
        It is the ratio of correctly predicted positive observations to the total predicted positive observations.

        ### Definition
        Precision = TP / (TP + FP)
        where
        - TP = True Positives
        - FP = False Positives

        ## Recall
        Also called sensitivity, recall is the ability of  aclassifier to find all the positive samples.
        Ratio of correctly predicted positive observations to all of the observations in the class.

        ### Definition:
        Recall = TP / (TP + FN)
        where
        - TP = True Positives
        - FN = False Negatives
        """

        y_true_all = []
        y_pred_all = []

        precisions = {}
        recalls = {}
        for cluster_id in self.processed_data["cluster_id"].unique():
            cluster_data = self.processed_data[self.processed_data["cluster_id"] == cluster_id]
            y_true = cluster_data["ground_truth_label"]
            y_pred = [cluster_id] * len(cluster_data)  # Predicted labels are the cluster ID

            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)

            precisions[cluster_id] = precision_score(y_true, y_pred, average="micro")
            recalls[cluster_id] = recall_score(y_true, y_pred, average="micro")

        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)

        # Generate the confusion matrix
        cm = confusion_matrix(y_true_all, y_pred_all)
        weighted_f1 = f1_score(y_true_all, y_pred_all, average="weighted")

        # Add precision and recall to the DataFrame
        self.processed_data["precision_for_cluster_id"] = self.processed_data["cluster_id"].apply(
            lambda x: precisions[x]
        )
        self.processed_data["recall_for_cluster_id"] = self.processed_data["cluster_id"].apply(
            lambda x: recalls[x]
        )

        print(f"Scores for {self.data_file_path} with linkage type: {self.linkage_type}")
        print(f"Head of processed data: \n{self.processed_data.head(20)}")
        print(
            f"Head of core fields: \n{self.processed_data[['ground_truth_label', 'cluster_id']].head(20)}"
        )
        print(f"Precision: {precisions}")
        print(f"Recalls: {recalls}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Weighted Average F1 Score: {weighted_f1}")

        if self.should_enforce_stopping_criteria or self.should_use_third_party:
            with open(self.output_txt_path, "w") as f:
                f.write(
                    f"Scores for {self.data_file_path} with linkage type: {self.linkage_type}\n"
                )
                f.write(f"Head of processed data: \n{self.processed_data.head(20)}\n")
                f.write(
                    f"Head of core fields: \n{self.processed_data[['ground_truth_label', 'cluster_id']].head(20)}\n"
                )
                f.write(f"Precision: {precisions}\n")
                f.write(f"Recalls: {recalls}\n")
                f.write(f"Confusion Matrix:\n{cm}\n")
                f.write(f"Weighted Average F1 Score: {weighted_f1}\n")
        return cm

    def compute(self) -> None:
        for step in self.steps:
            step()


if __name__ == "__main__":
    permutations: list[HACAlgoInput] = [
        # Third Party
        HACAlgoInput(DATA_FILE_PATH_EASY, True, False, LinkageType.SINGLE),
        HACAlgoInput(DATA_FILE_PATH_EASY, True, False, LinkageType.COMPLETE),
        HACAlgoInput(DATA_FILE_PATH_EASY, True, False, LinkageType.AVERAGE),
        HACAlgoInput(DATA_FILE_PATH_EASY, True, False, LinkageType.CENTROID),
        HACAlgoInput(DATA_FILE_PATH_EASY, True, False, LinkageType.WARD),
        HACAlgoInput(DATA_FILE_PATH_CPX, True, False, LinkageType.SINGLE),
        HACAlgoInput(DATA_FILE_PATH_CPX, True, False, LinkageType.COMPLETE),
        HACAlgoInput(DATA_FILE_PATH_CPX, True, False, LinkageType.AVERAGE),
        HACAlgoInput(DATA_FILE_PATH_CPX, True, False, LinkageType.CENTROID),
        HACAlgoInput(DATA_FILE_PATH_CPX, True, False, LinkageType.WARD),
        # From scratch - easy
        HACAlgoInput(DATA_FILE_PATH_EASY, False, True, LinkageType.SINGLE),
        HACAlgoInput(DATA_FILE_PATH_EASY, False, True, LinkageType.COMPLETE),
        HACAlgoInput(DATA_FILE_PATH_EASY, False, True, LinkageType.AVERAGE),
        HACAlgoInput(DATA_FILE_PATH_EASY, False, False, LinkageType.SINGLE),
        HACAlgoInput(DATA_FILE_PATH_EASY, False, False, LinkageType.COMPLETE),
        HACAlgoInput(DATA_FILE_PATH_EASY, False, False, LinkageType.AVERAGE),
        # From scratch - hard
        HACAlgoInput(DATA_FILE_PATH_CPX, False, True, LinkageType.SINGLE),
        HACAlgoInput(DATA_FILE_PATH_CPX, False, True, LinkageType.COMPLETE),
        HACAlgoInput(DATA_FILE_PATH_CPX, False, True, LinkageType.AVERAGE),
        HACAlgoInput(DATA_FILE_PATH_CPX, False, False, LinkageType.SINGLE),
        HACAlgoInput(DATA_FILE_PATH_CPX, False, False, LinkageType.COMPLETE),
        HACAlgoInput(DATA_FILE_PATH_CPX, False, False, LinkageType.AVERAGE),
    ]
    for permutation in permutations:
        hac = HierarchicalAgglomerativeClustering(
            data_file_path=permutation.data_file_path,
            linkage_type=permutation.linkage_type,
            should_enforce_stopping_criteria=permutation.should_stop_early,
            should_use_third_party=permutation.should_use_third_party,
        )
        hac.compute()

    # hac = HierarchicalAgglomerativeClustering(
    #     linkage_type=LinkageType.SINGLE,
    #     should_enforce_stopping_criteria=SHOULD_STOP_EARLY,
    #     should_use_third_party=SHOULD_USE_THIRD_PARTY,
    # )
    # hac.compute()
