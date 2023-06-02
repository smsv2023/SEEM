import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances_argmin_min
from shapely.geometry import LineString
from scipy.spatial.distance import pdist

# find out the longest distance of two pixels in the masked area
def mask_size(mask):
    # Assuming 'mask' is your binary mask image where object pixels are 1
    y, x = np.where(mask == 1)

    # Stack x and y coordinates together
    coords = np.stack((x, y), axis=-1)

    # Compute pairwise distances
    distances = pdist(coords)

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
    
def find_edge_candidates(lines):
    orientations = find_orientations(lines)
    # Use DBSCAN to cluster the lines based on their orientations
    dbscan = DBSCAN(eps=np.pi/32, min_samples=5).fit(orientations)  # adjust the parameters as needed
    labels = dbscan.labels_
    
    # Compute the average orientation of each cluster
    clusters = np.unique(labels)
    avg_orientations = np.array([orientations[labels == i].mean() for i in clusters])
    
    # Step 2: Rating Clusters
    ratings = np.zeros(len(clusters))
    for i in clusters:
        # Criterion 1: Contains more long lines
        group_lines = lines[labels == i]
        group_lengths = np.sqrt((group_lines[:, 2] - group_lines[:, 0])**2 + (group_lines[:, 3] - group_lines[:, 1])**2)
        ratings[i] += group_lengths.sum()

        # Criterion 2: More perpendicular to any high rating other groups
        diff = np.abs(avg_orientations[i] - avg_orientations)
        diff = np.min(diff, np.pi - diff)
        ratings[i] += np.sum(diff > np.pi/2 - np.pi/8)  # adjust the tolerance as needed

        # Criterion 3: Long line are more close the edge of the bounding box of the line cluster
        bbox = np.vstack([group_lines[:, :2], group_lines[:, 2:]]).min(axis=0), np.vstack([group_lines[:, :2], group_lines[:, 2:]]).max(axis=0)
        distances, _ = pairwise_distances_argmin_min(group_lines.reshape(-1, 2), np.array(bbox))
        ratings[i] += np.sum(distances < 10)  # adjust the threshold as needed

        # Criterion 4: Contains lines with starting/ending points close to starting/ending points of other high rating cluster
        # You'll need to implement this part based on your specific requirements
        for j in clusters:
            if i < j:
                group_lines_j = lines[labels == j]
                for line_i in group_lines:
                    for line_j in group_lines_j:
                        distances1, distances2 = lines_intersect(line_i, line_j)
                        if np.any(distances_i < 10) and np.any(distances_j < 10):  # adjust the threshold as needed
                            ratings[i] += 1
                            ratings[j] += 1
                            
    # Select the top 2 rated clusters
    top_clusters = np.argsort(ratings)[-2:]
    return top_clusters

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
mask = cv2.imread(os.path.join(folder, mask_file))
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
