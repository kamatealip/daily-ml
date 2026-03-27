from math import dist
from typing import List, Tuple

Point = Tuple[float, float]


def single_link_distance(cluster_a: List[int], cluster_b: List[int], points: List[Point]) -> float:
    return min(dist(points[i], points[j]) for i in cluster_a for j in cluster_b)


def hierarchical_clustering(points: List[Point], target_clusters: int) -> List[List[int]]:
    if target_clusters < 1 or target_clusters > len(points):
        raise ValueError("target_clusters must be between 1 and the number of points")

    clusters = [[index] for index in range(len(points))]
    step = 1

    while len(clusters) > target_clusters:
        best_pair = (0, 1)
        best_distance = float("inf")

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = single_link_distance(clusters[i], clusters[j], points)
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (i, j)

        left_index, right_index = best_pair
        merged_cluster = sorted(clusters[left_index] + clusters[right_index])

        print(
            f"Step {step}: merge {clusters[left_index]} and {clusters[right_index]} "
            f"(distance = {best_distance:.2f})"
        )

        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in best_pair
        ]
        clusters.append(merged_cluster)
        step += 1

    return sorted(clusters, key=lambda cluster: cluster[0])


def print_clusters(clusters: List[List[int]], points: List[Point]) -> None:
    print("\nFinal clusters:")
    for number, cluster in enumerate(clusters, start=1):
        members = [points[index] for index in cluster]
        print(f"Cluster {number}: {members}")


def main() -> None:
    points: List[Point] = [
        (1.0, 1.0),
        (1.5, 1.2),
        (2.0, 1.8),
        (8.0, 8.0),
        (8.5, 8.2),
        (9.0, 8.8),
        (5.0, 2.0),
        (5.5, 2.5),
    ]

    print("Points:")
    for index, point in enumerate(points):
        print(f"{index}: {point}")

    clusters = hierarchical_clustering(points, target_clusters=3)
    print_clusters(clusters, points)


if __name__ == "__main__":
    main()
