import random
import sys
import matplotlib.pyplot as plt
import math

FIG_TITLE = 'clustering'
MAX_LIMIT_CANVAS = 5000
MAX_RANDOM_OFFSET = 100

INNITIAL_SIZE = 20
ADDITIONAL_SIZE = 1000
NUMBER_OF_CLUSTERS = 10
CHECK_THRESHOLD = 500

DEBUG = True
if DEBUG:
    POINTS_PATH = 'debug_points.txt'
    NUMBER_OF_CLUSTERS = 3
else:
    POINTS_PATH = 'random_points.txt'

# HELPER
def generate_points(initial_size, additional_size, file_path):
    initial_points = set()
    x = set()
    y = set()

    while len(initial_points) < initial_size:
        new_x = random.uniform(-MAX_LIMIT_CANVAS, MAX_LIMIT_CANVAS)
        new_y = random.uniform(-MAX_LIMIT_CANVAS, MAX_LIMIT_CANVAS)

        if new_x not in x and new_y not in y:
            initial_points.add((new_x, new_y))
            x.add(new_x)
            y.add(new_y)

    additional_points = set()

    while len(additional_points) < additional_size:
        selected_point = random.choice(list(additional_points.union(initial_points)))
        x_offset, y_offset = random.uniform(-MAX_RANDOM_OFFSET, MAX_RANDOM_OFFSET), random.uniform(-MAX_RANDOM_OFFSET, MAX_RANDOM_OFFSET)

        new_x, new_y = selected_point[0] + x_offset, selected_point[1] + y_offset

        new_x = max(-MAX_LIMIT_CANVAS, min(MAX_LIMIT_CANVAS, new_x))
        new_y = max(-MAX_LIMIT_CANVAS, min(MAX_LIMIT_CANVAS, new_y))

        if new_x not in x and new_y not in y:
            additional_points.add((new_x, new_y))
            x.add(new_x)
            y.add(new_y)

    all_points = additional_points.union(initial_points)

    try: 
        with open(file_path, 'w') as file:
            for point in all_points:
                file.write(f"{point[0]} {point[1]}\n")
    except IOError:
        sys.exit()
def read_points_from_file(file_path) -> list:
    points = []
    try:
        with open(file_path) as f:
            for line in f:
                tmp = line.strip().split()
                points.append([eval(tmp[0]), eval(tmp[1])])
            return points
    except IOError:
        sys.exit()
def draw(clusters, centroids):
    fig, axs = plt.subplots()
    plt.suptitle('Clustering')
    plt.xlim(-MAX_LIMIT_CANVAS, MAX_LIMIT_CANVAS)
    plt.ylim(-MAX_LIMIT_CANVAS, MAX_LIMIT_CANVAS)


    cur_marker = 0
    markers = [(random.random(), random.random(), random.random()) for _ in range(len(clusters))]

    for cluster in clusters:
        for point in cluster:
            axs.plot(point[0], point[1], c=markers[cur_marker], marker="o")
        cur_marker += 1
    
    for centroid in centroids:
        axs.plot(centroid[0], centroid[1], markersize=10, c="black", marker="x")
    plt.savefig(FIG_TITLE)
def print_matrix(distance_matrix):
    for row in distance_matrix:
        for distance in row:
            print("{:.2f}".format(distance), end="\t")
        print()

# CHECK
def calculate_cluster_efficiency(clusters, centroids, distance_threshold=CHECK_THRESHOLD):
    successful_matches = 0

    for cluster, centroid in zip(clusters, centroids):
        distances = [calculate_distance(point, centroid) for point in cluster]
        check = all(dist < distance_threshold for dist in distances)

        if check:
            successful_matches += 1

    print(f"accuracy: {successful_matches / len(clusters) * 100} %")

# ALGO HELPER
def create_distance_matrix(points) -> list:
    num_points = len(points)
    distance_matrix = []

    for i in range(num_points):
        first_point = points[i]
        distances_from_first_point = []

        for j in range(i + 1): # LOWER TRIANGLE ONLY
            second_point = points[j]
            distance = calculate_distance(first_point, second_point)
            distances_from_first_point.append(distance)

        distance_matrix.append(distances_from_first_point)

    return distance_matrix
def find_shortest_distance_indices(matrix):
    min_value = None
    min_row, min_col = -1, -1

    num_rows = len(matrix)

    for i in range(num_rows):
        for j in range(i):  # LOWER TRIANGLE ONLY
            if min_value is None or matrix[i][j] < min_value:
                min_value = matrix[i][j]
                min_row, min_col = i, j

    return min_row, min_col
def calculate_distance(first, second) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])

# CLUSTERING
def calculate_centroid(points) -> list:
    sum_x, sum_y = 0, 0
    for x, y in points:
        sum_x += x
        sum_y += y

    centroid_x = sum_x / len(points)
    centroid_y = sum_y / len(points)
    return [centroid_x, centroid_y]

def calculate_medoid(points) -> list:
    centroid = calculate_centroid(points)
    distances = [calculate_distance(centroid, point) for point in points]
    index = min(range(len(distances)), key=distances.__getitem__)
    return [points[index][0], points[index][1]]

def clustering_centroid(points, k):
    clusters = [[point] for point in points]
    centers = [point for point in points]
    matrix = create_distance_matrix(points)
    matrix_len = len(matrix)

    if DEBUG:
        print_matrix(matrix)
        print("FIRST\n\n")

    while len(clusters) > k:
        # add nearest element to the nearest cluster
        min_row, min_col = find_shortest_distance_indices(matrix)
        clusters[min_row] += clusters[min_col]
        
        # update the center of the cluster
        centers[min_row] = calculate_centroid(clusters[min_row])
        
        # remove element from centers, and its single clusters
        centers.pop(min_col)
        clusters.pop(min_col)
       
        # remove element from matrix
        for i in range(matrix_len):
            if min_col < i+1:
                matrix[i].pop(min_col)
        matrix.pop(min_col)
        matrix_len -= 1

        # update matrix
        for i in range(matrix_len):
            if min_col < i+1:
                matrix[i][min_col] = calculate_distance(centers[min_col] , centers[i])
            if i < min_col:
                matrix[min_col][i] = calculate_distance(centers[min_col], centers[i])
        
        if DEBUG:
            print()
            print_matrix(matrix)
            print(f"CLUSTEERS{clusters}")
            print(f"CENTERS{centers}\n\n")
        
    return clusters, centers
def clustering_medoid(points, k):
    clusters = [[point] for point in points]
    centers = [point for point in points]

    matrix = create_distance_matrix(points)
    matrix_len = len(matrix)

    if DEBUG:
        print_matrix(matrix)
        print("FIRST\n\n")

    while len(clusters) > k:
        # add nearest element to the nearest cluster
        min_row, min_col = find_shortest_distance_indices(matrix)
        clusters[min_row] += clusters[min_col]
        
        # update the center of the cluster
        centers[min_row] = calculate_medoid(clusters[min_row])
        
        # remove element from centers, and its single clusters
        centers.pop(min_col)
        clusters.pop(min_col)
       
        # remove element from matrix
        for i in range(matrix_len):
            if min_col < i+1:
                matrix[i].pop(min_col)
        matrix.pop(min_col)
        matrix_len -= 1

        # update matrix
        for i in range(matrix_len):
            if min_col < i+1:
                matrix[i][min_col] = calculate_distance(centers[min_col] , centers[i])
            if i < min_col:
                matrix[min_col][i] = calculate_distance(centers[min_col], centers[i])
        
        if DEBUG:
            print()
            print_matrix(matrix)
            print(f"CLUSTEERS{clusters}")
            print(f"CENTERS{centers}\n\n")
        
    return clusters, centers

def clustering(points, k, center_centroid):
    if center_centroid:
        return clustering_centroid(points, k)
    return  clustering_medoid(points, k)


def main():
    if DEBUG:
        points = read_points_from_file(POINTS_PATH)
        clusters, centroids = clustering(points, NUMBER_OF_CLUSTERS, True)
        calculate_cluster_efficiency(clusters, centroids)
    
    
    else: 
        generate_points(INNITIAL_SIZE, ADDITIONAL_SIZE, POINTS_PATH)
        points = read_points_from_file(POINTS_PATH)
        clusters, centroids = clustering(points, NUMBER_OF_CLUSTERS, True)
        calculate_cluster_efficiency(clusters, centroids)
        draw(clusters, centroids)
    

if __name__ == "__main__":
    main()