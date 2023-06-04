import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min
from shapely.geometry import LineString
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull

# find out the longest distance of two pixels in the masked area
def find_mask_size(mask):
    # Assuming 'mask' is your binary mask image where object pixels are 1
    y, x = np.where(mask == 1)

    # Stack x and y coordinates together
    coords = np.stack((x, y), axis=-1)
    
    # Compute the convex hull
    hull = ConvexHull(coords)

    # Get the coordinates of the hull vertices
    hull_coords = coords[hull.vertices]

    # Compute pairwise distances
    distances = pdist(hull_coords)

    # Find the maximum distance
    max_distance = np.max(distances)    
    return max_distance

def canny_detection(gray):
    # Perform Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    return edges

def show_canny(image, edges):
    # Display Canny Edge Detection Image
    cv2.imshow('Canny Edge Detection', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Probabilistic Hough
def detect_line(gray): 
    # Blur the image for better edge detection
    #img_blur = cv2.GaussianBlur(gray, (3,3), 0) 
    
    # Perform Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Perform Canny Edge Detection
    #edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
    
    # Perform Probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
    return lines

def show_lines(image, lines):
    color = 'b'
    # Plot each line with color
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line
        # Choose color based on cluster label
        plt.plot((x1, x2), (y1, y2), color=color)
        # Add text to indicate the cluster number
        plt.text(x1, y1, f'{i}', color=color)

    # plt's origin is at left bottom, so flip it vertically
    plt.gca().invert_yaxis()
    plt.imshow(image)
    plt.show()

def line_to_center_direction(line):
    center = np.array([(line[0] + line[2]) / 2, (line[1] + line[3]) / 2])  
    direction = np.array([line[2] - line[0], line[3] - line[1]], dtype=float)
    direction /= np.linalg.norm(direction)
    return center, direction
    
def line_distance(line1, line2, width, height):
    center1, direction1 = line_to_center_direction(line1)
    center2, direction2 = line_to_center_direction(line2)

    dot_product = np.dot(direction1, direction2)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Clip to the range [-1, 1]
    direction_difference = np.arccos(dot_product)
    
    # Threshold for direction difference
    direction_threshold = np.pi / 180  # 1 degree
    

    if direction_difference < direction_threshold:
        # If the lines are roughly parallel, move line2 to be closer to line1
        # Move the second line along its direction so that its center is closest to the first line's center
        t = np.dot(center1 - center2, direction1) / np.dot(direction1, direction1)
        center2_moved = center2 + t * direction1
        center_distance_moved = np.linalg.norm(center2_moved - center1)
        
        # Calculate the diagonal length of the image
        image_diagonal = np.sqrt(width**2 + height**2)

        # Normalize the center_distance_moved by the image diagonal
        center_distance_moved_normalized = center_distance_moved / image_diagonal

        return center_distance_moved_normalized
    else:
        #center_distance = np.linalg.norm(center2 - center1)
        # If the lines are not parallel, return a large distance
        return np.inf


# define decorator to be used by DBSCAN clustering
def line_distance_decorator(width, height):
    def line_distance_decorated(line1, line2):
        return line_distance(line1, line2, width, height)
    return line_distance_decorated

# use a clustering algorithm like DBSCAN, which can group together 
# line segments that are close in terms of both distance and orientation. 
def cluster_lines(lines, width, height):
    # Create an array where each row represents a line segment
    # The columns could be the coordinates of the center point and the orientation
    # lines_array = np.array([[(line[0][0]+line[0][2])/2, (line[0][1]+line[0][3])/2, np.arctan2(line[0][3]-line[0][1], line[0][2]-line[0][0])] for line in lines])
    
    #lines_array = np.array([[x_center, y_center, orientation] for line in lines])

    # Apply DBSCAN
    #clustering = DBSCAN(eps=0.3, min_samples=2).fit(lines_array)
    # use customer distance for the polar system lines
    #polar_lines = convert_to_polar_lines(lines)
    # default eps is 0.5, use 5 to get more clusters
    #clustering = DBSCAN(eps=5, min_samples=2, metric=hough_distance).fit(polar_lines)
    #clustering = DBSCAN(eps=5, min_samples=2, metric=line_distance).fit(lines)
    
    # Decorate the line_distance function with the image width and height
    line_distance_metric = line_distance_decorator(width, height)
    # Use the decorated function as the metric function
    # define eps:
    max_center_distance = 20 # assume 4000 width for 4m object, 20 pixels is about 20mm
    eps = max_center_distance/np.sqrt(width**2 + height**2)
    clustering = DBSCAN(eps, min_samples=2, metric=line_distance_metric).fit(lines)

    # The labels_ attribute contains the cluster labels for each line segment
    labels = clustering.labels_
    return clustering

def show_clusters(image, lines, labels):
    # Create a list of colors for each cluster
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']    
    # Plot each line with color corresponding to its cluster
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line
        # Choose color based on cluster label
        color = colors[labels[i] % len(colors)]
        plt.plot((x1, x2), (y1, y2), color=color)
        # Add text to indicate the cluster number
        if labels[i] != -1:
            plt.text(x1, y1, f'{labels[i]}', color=color)

    # plt's origin is at left bottom, so flip it vertically
    plt.gca().invert_yaxis()
    plt.imshow(image)
    plt.show()

def find_representative_line(cluster_lines):
    # Compute the direction of the line
    dx = np.mean(cluster_lines[:, 2] - cluster_lines[:, 0])
    dy = np.mean(cluster_lines[:, 3] - cluster_lines[:, 1])

    # Normalize the direction
    length = np.sqrt(dx**2 + dy**2)
    dx /= length
    dy /= length

    # Compute the mean point of the line segments
    mean_point = np.mean(cluster_lines[:, :2], axis=0)

    # Project the start and end points onto the direction
    projections = (cluster_lines[:, [0, 2]] - mean_point[0]) * dx + (cluster_lines[:, [1, 3]] - mean_point[1]) * dy

    # Find the minimum and maximum projections
    min_proj = np.min(projections)
    max_proj = np.max(projections)

    # Compute the start and end points of the representative line
    start = np.round(mean_point + np.array([min_proj * dx, min_proj * dy])).astype(int)
    end = np.round(mean_point + np.array([max_proj * dx, max_proj * dy])).astype(int)


    # Create a new line with these points
    representative_line = np.concatenate([start, end])    
    return representative_line

def find_representative_lines(lines, labels):
    # Initialize an empty list to hold the representative lines
    representative_lines = []

    # For each cluster label...
    for label in set(labels):
        # Get the lines in this cluster
        cluster_lines = lines[labels == label]
        if label!=-1:
            representative_line = find_representative_line(cluster_lines)
            # Add this line to the list of representative lines
            representative_lines.append(representative_line)
        else:
            for line in cluster_lines:
                representative_lines.append(line)

    # Convert the list to a numpy array
    representative_lines = np.array(representative_lines)
    return representative_lines
   
def find_orientations(lines):
    # Compute the orientations of the lines
    lines = np.array(lines)  # shape: [n_lines, 4]
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    orientations = np.arctan2(dy, dx).reshape(-1, 1)
    return orientations

# checks if two lines intersect, and return the distance of end points to the intersection
def lines_intersect(line1, line2):
    line1_geom = LineString([(line1[0], line1[1]), (line1[2], line1[3])])
    line2_geom = LineString([(line2[0], line2[1]), (line2[2], line2[3])])
    intersection = line1_geom.intersection(line2_geom)
    if intersection.is_empty:
        return np.inf
    intersection_point = np.array([intersection.x, intersection.y])
    #distances1 = np.sqrt((group_lines_i[:, [0, 2]] - intersection_point[0])**2 + (group_lines_i[:, [1, 3]] - intersection_point[1])**2)
    #distances2 = np.sqrt((group_lines_j[:, [0, 2]] - intersection_point[0])**2 + (group_lines_j[:, [1, 3]] - intersection_point[1])**2)
    distances1 = np.sqrt((line1[0, 2] - intersection_point[0])**2 + (line1[1, 3] - intersection_point[1])**2)
    distances2 = np.sqrt((line2[0, 2] - intersection_point[0])**2 + (line2[1, 3] - intersection_point[1])**2)
    return distances1, distances2
    
def find_top_clusters(lines, angle_threshold=5, length_threshold=100, close_threshold=20):
    orientations = find_orientations(lines)

    # Use DBSCAN to cluster the lines based on their orientations
    dbscan = DBSCAN(eps=np.pi/32, min_samples=5).fit(orientations)  # adjust the parameters as needed
    labels = dbscan.labels_
    
    # Step 2: Rating Clusters
    clusters = np.unique(labels)        
    ratings = np.zeros(len(clusters))
    for i in clusters:
        # Criterion 1: Contains long lines
        group_lines = lines[labels == i]
        # Compute the length of each line
        group_lengths = np.sqrt((group_lines[:, 0] - group_lines[:, 2])**2 + (group_lines[:, 1] - group_lines[:, 3])**2)
        # Only consider lines that are long enough
        long_lines = group_lines[lengths > length_threshold]  # adjust the threshold as needed
        ratings[i] += 1 if lenth(long_lines)>0 else 0

        # Criterion 2: line closer the edge of the convex hull of the line cluster
        for i in clusters:
            group_lines = lines[labels == i]
            group_points = group_lines.reshape(-1, 2)
            hull = ConvexHull(group_points)
            hull_polygon = Polygon(group_points[hull.vertices])
            for line in group_lines:
                line_geom = LineString([(line[0], line[1]), (line[2], line[3])])
                min_distance = hull_polygon.boundary.distance(line_geom)
                if min_distance < close_threshold:  # adjust the threshold as needed
                    ratings[i] += 1

        # Criterion 3: Contains lines with starting/ending points close to starting/ending points of other high rating cluster
        # You'll need to implement this part based on your specific requirements
        for j in clusters:
            if i < j:
                group_lines_j = lines[labels == j]
                for line_i in group_lines:
                    for line_j in group_lines_j:
                        distances1, distances2 = lines_intersect(line_i, line_j)
                        if np.any(distances_i < 10) and np.any(distances_j < close_threshold):  # adjust the threshold as needed
                            ratings[i] += 1
                            ratings[j] += 1

    # Get the indices of the clusters sorted by their ratings
    sorted_clusters = np.argsort(ratings)[::-1]

    # Exclude clusters with vertical orientation
    horizontal_clusters = []
    for i in sorted_clusters:
        group_lines = lines[labels == i]
        angles = np.abs(np.arctan2(group_lines[:,3] - group_lines[:,1], group_lines[:,2] - group_lines[:,0]))
        median_angle = np.median(angles)
        if np.abs(np.pi/2 - median_angle) < np.pi/(180/angle_threshold):  # angle_threshold is 5 degree by default
            horizontal_clusters.append(i)
    vertical_clusters = [i for i in sorted_clusters if i not in horizontal_clusters]
            
    # Ensure top two clusters have different orientations
    top_clusters = []
    for i in horizontal_clusters:
        if len(top_clusters) == 0:
            top_clusters.append(i)
        else:
            group1_lines = lines[labels == top_clusters[0]]
            group2_lines = lines[labels == i]
            angles1 = np.abs(np.arctan2(group1_lines[:,3] - group1_lines[:,1], group1_lines[:,2] - group1_lines[:,0]))
            angles2 = np.abs(np.arctan2(group2_lines[:,3] - group2_lines[:,1], group2_lines[:,2] - group2_lines[:,0]))
            median_angle1 = np.median(angles1)
            median_angle2 = np.median(angles2)
            if np.abs(median_angle1 - median_angle2) > np.pi/6:  # adjust the threshold as needed
                top_clusters.append(i)
        if len(top_clusters) == 2:
            break
            
    # Select the top 2 rated clusters
    # top_clusters = np.argsort(ratings)[-2:]
    return top_clusters, vertical_clusters[0]

# for each cluster, find the upper/lower most lines
# the upper most lines will be the candiate edge of the further sides
# need to move all the lines along their direction to have same center 'x'
def find_upper_lower_most_lines(lines, labels):
    uppermost_lines = []
    lowermost_lines = []
    clusters = np.unique(labels)
    for i in clusters:
        group_lines = lines[labels == i]
        # Compute the center point of each line
        centers = np.mean(group_lines[:, [0, 2]], axis=1), np.mean(group_lines[:, [1, 3]], axis=1)
        # Compute the direction of each line
        directions = np.arctan2(group_lines[:, 3] - group_lines[:, 1], group_lines[:, 2] - group_lines[:, 0])
        # Move each line along its direction so that its center point has the same x-coordinate
        avg_x = np.mean(centers[0])
        moved_centers = centers[0] + (avg_x - centers[0]) * np.cos(directions), centers[1] + (avg_x - centers[0]) * np.sin(directions)
        # The line with the smallest y-coordinate is the uppermost
        uppermost_line_index = np.argmin(moved_centers[1])
        lowermost_line_index = np.argmax(moved_centers[1])
        uppermost_lines.append(group_lines[uppermost_line_index])
        lowermost_lines.append(group_lines[lowermost_line_index])
    return uppermost_lines, lowermost_lines

# for each cluster, find the lower most pair of parallel lines
# the upper most lines will be the candiate edge of the further sides
# need to move all the lines along their direction to have same center 'x'
def find_lowest_parallel_pairs(lines, labels):
    lowermost_pairs = []
    clusters = np.unique(labels)
    for i in clusters:
        group_lines = lines[labels == i]
        # Compute the length of each line
        lengths = np.sqrt((group_lines[:, 0] - group_lines[:, 2])**2 + (group_lines[:, 1] - group_lines[:, 3])**2)
        # Only consider lines that are long enough
        long_lines = group_lines[lengths > length_threshold]  # adjust the threshold as needed
        # Compute the slope of each line
        slopes = (long_lines[:, 3] - long_lines[:, 1]) / (long_lines[:, 2] - long_lines[:, 0] + 1e-8)
        # Sort the lines by their slope
        sorted_indices = np.argsort(slopes)
        sorted_lines = long_lines[sorted_indices]
        # Find the pair of lines with the smallest difference in slope
        min_diff = np.inf
        lowermost_pair = None
        for j in range(len(sorted_lines) - 1):
            diff = np.abs(slopes[sorted_indices[j+1]] - slopes[sorted_indices[j]])
            if diff < min_diff:
                # Compute the center point of each line
                centers = np.mean(sorted_lines[[j, j+1], [0, 2]], axis=1), np.mean(sorted_lines[[j, j+1], [1, 3]], axis=1)
                # Compute the direction of each line
                directions = np.arctan2(sorted_lines[[j, j+1], 3] - sorted_lines[[j, j+1], 1], sorted_lines[[j, j+1], 2] - sorted_lines[[j, j+1], 0])
                # Move each line along its direction so that its center point has the same x-coordinate
                avg_x = np.mean(centers[0])
                moved_centers = centers[0] + (avg_x - centers[0]) * np.cos(directions), centers[1] + (avg_x - centers[0]) * np.sin(directions)
                # The pair with the largest y-coordinate is the lowermost
                if np.max(moved_centers[1]) < min_diff:
                    min_diff = np.max(moved_centers[1])
                    lowermost_pair = sorted_lines[[j, j+1]]
            lowermost_pairs.append(lowermost_pair)
    return lowermost_pairs

# not finished yet
def filter_lines(lines, angle_threshold, length_threshold):
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate the angle of the line to the vertical
        angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi
        if angle < 0:
            angle += 180

        # Calculate the length of the line
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Apply the thresholds
        if abs(angle - 90) <= angle_threshold and length >= length_threshold:
            filtered_lines.append(line)

    return filtered_lines
        
# Load the image
folder ='/Users/seanmao/Pictures/SEEM/output/Test001'
file = '10_dining table_cropped.png'
mask_file= '10_dining table_mask_resized.png'
image = cv2.imread(os.path.join(folder, file))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
mask = cv2.imread(os.path.join(folder, mask_file), cv2.IMREAD_GRAYSCALE)
lines = detect_line(gray)
lines = lines.reshape(-1, 4)
clustering = cluster_lines(lines, image.shape[1], image.shape[0])
labels = clustering.labels_
representative_lines=find_representative_lines(lines, labels)
show_lines(image, representative_lines)
#show_clusters(image, lines, labels)
#show_lines(image, lines)
#key_points = ORB(gray)
#show_ORB(image, key_points)
#edges=canny_detection(gray)
#show_canny(image, edges)
#sobelx, sobely, sobelxy = sobel_detection(gray)
#show_sobel_edge(image, sobelx, sobely, sobelxy)
#dst = detect_corner_points(gray)
#show_corners(image, dst)
