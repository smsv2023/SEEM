import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

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
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    return lines

def show_lines(image, lines):
    # Draw the lines on the original image
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Show the image
    cv2.imshow('Image with lines', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# minimize the distance between the centers of the two lines while keeping them parallel, 
# which should give a better measure of their similarity.
# problem: If the two lines have significantly different directions, then they are not on 
# the same line, then moving the line doesn't make sense
def line_distance(line1, line2, width, height):
    # Calculate the centers of the lines
    center1 = np.array([(line1[0] + line1[2]) / 2, (line1[1] + line1[3]) / 2])
    center2 = np.array([(line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2])

    # Calculate the direction of the lines
    direction1 = np.array([line1[2] - line1[0], line1[3] - line1[1]], dtype=float)
    direction2 = np.array([line2[2] - line2[0], line2[3] - line2[1]], dtype=float)

    # Normalize the directions
    direction1 /= np.linalg.norm(direction1)
    direction2 /= np.linalg.norm(direction2)

    # Move the second line along its direction so that its center is closest to the first line's center
    t = np.dot(center1 - center2, direction1) / np.dot(direction1, direction1)
    center2_moved = center2 + t * direction1

    # Calculate the distance between the centers of the lines and the difference in their direction
    center_distance = np.linalg.norm(center1 - center2_moved)
    cos_angle = np.dot(direction1, direction2)
    cos_angle = np.clip(cos_angle, -1, 1)
    direction_difference = np.arccos(cos_angle)

    # Normalize center distance and direction difference
    center_distance_normalized = center_distance / np.sqrt(width**2 + height**2)
    direction_difference_normalized = direction_difference / np.pi

    # Define weights
    center_distance_weight = 0.5
    direction_difference_weight = 1
    
    # Calculate final distance
    distance = center_distance_weight * center_distance_normalized + direction_difference_weight * direction_difference_normalized

    return distance

def line_to_center_direction(line):
    center = np.array([(line[0] + line[2]) / 2, (line[1] + line[3]) / 2])  
    direction = np.array([line[2] - line[0], line[3] - line[1]], dtype=float)
    direction /= np.linalg.norm(direction)
    return center, direction
    
def line_distance(line1, line2):
    center1, direction1 = line_to_center_direction(line1)
    center2, direction2 = line_to_center_direction(line2)

    direction_difference = np.arccos(np.dot(direction1, direction2))
    center_distance = np.linalg.norm(center2 - center1)

    # Threshold for direction difference
    direction_threshold = np.pi / 180  # 30 degrees

    if direction_difference < direction_threshold:
        # If the lines are roughly parallel, move line2 to be closer to line1
        center2_moved = center2 + (center1 - center2) * direction2
        center_distance_moved = np.linalg.norm(center2_moved - center1)
        return center_distance_moved
    else:
        # If the lines are not parallel, return a large distance
        return np.inf


# define decorator to be used by DBSCAN clustering
def line_distance_decorator(width, height):
    def line_distance_decorated(line1, line2):
        return line_distance(line1, line2, width, height)
    return line_distance_decorated

# use a clustering algorithm like DBSCAN, which can group together 
# line segments that are close in terms of both distance and orientation. 
def cluster_lines(lines):
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
    max_center_distance = 10 # assume 4000 width for 4m object, 10 pixels is about 10mm
    max_angle_distance = np.pi/180 # assume the angle difference is smaller than 1 degree, 
    eps = max_center_distance/np.sqrt(width**2 + height**2) * 0.5 + max_angle_distance/np.pi
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
        representative_line = find_representative_line(cluster_lines)

        # Add this line to the list of representative lines
        representative_lines.append(representative_line)

    # Convert the list to a numpy array
    representative_lines = np.array(representative_lines)
    return representative_lines
   
        
# Load the image
folder ='/Users/seanmao/Pictures/SEEM/output/Test001'
file = '10_dining table_cropped.png'
image = cv2.imread(os.path.join(folder, file))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
lines = detect_line(gray)
lines = lines.reshape(-1, 4)
clustering = cluster_lines(lines)
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
