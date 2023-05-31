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

def line_distance(line1, line2):
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

    # Combine the two distances into a single measure
    distance = center_distance + direction_difference

    return distance

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
    clustering = DBSCAN(eps=5, min_samples=2, metric=line_distance).fit(lines)

    # The labels_ attribute contains the cluster labels for each line segment
    labels = clustering.labels_
    return clustering

def show_clusters(lines, labels):
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
    plt.show()

def find_representative_lines(lines, labels):
    # Initialize an empty list to hold the representative lines
    representative_lines = []

    # For each cluster label...
    for label in set(labels):
        # Get the lines in this cluster
        cluster_lines = lines[labels == label]

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
#show_clusters(lines, labels)
#show_lines(image, lines)
#key_points = ORB(gray)
#show_ORB(image, key_points)
#edges=canny_detection(gray)
#show_canny(image, edges)
#sobelx, sobely, sobelxy = sobel_detection(gray)
#show_sobel_edge(image, sobelx, sobely, sobelxy)
#dst = detect_corner_points(gray)
#show_corners(image, dst)
