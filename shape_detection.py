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
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Show the image
    cv2.imshow('Image with lines', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Sobel Edge Detection
def sobel_detection(gray):
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(gray, (3,3), 0) 

    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
    return sobelx, sobely, sobelxy

def show_sobel_edge(image, sobelx, sobely, sobelxy):
    # Display Sobel Edge Detection Images
    cv2.imshow('Sobel X', sobelx)
    cv2.waitKey(0)
    cv2.imshow('Sobel Y', sobely)
    cv2.waitKey(0)
    cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
    cv2.waitKey(0)


def detect_corner_points(gray):
    # Convert to float
    gray = np.float32(gray)

    # Apply Harris Corner detection
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # Dilate to mark the corners
    dst = cv2.dilate(dst, None)
    return dst

def show_corners(image, dst):
    '''
    # Threshold for an optimal value, it may vary depending on the image
    image[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imshow('Harris Corners', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

# ORB (Oriented FAST and Rotated BRIEF) to detect key points in the image. 
def ORB(gray):
    # Create an ORB object
    orb = cv2.ORB_create()

    # Detect the key points
    key_points = orb.detect(image, None)

    # Compute the descriptors
    key_points, descriptors = orb.compute(image, key_points)
    return key_points

def show_ORB(image, key_points):
    # Draw the key points on the image
    image_with_key_points = cv2.drawKeypoints(image, key_points, None, color=(0, 255, 0), flags=0)

    # Display the image
    plt.imshow(image_with_key_points), plt.show()

# convert lines from Probabilistic Hough Transform with starting/ending points to lines in the polar coordinate system
def convert_to_polar_lines(lines):
    polar_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate theta
        theta = np.arctan2(y2 - y1, x2 - x1)

        # Calculate rho
        rho = x1 * np.cos(theta) + y1 * np.sin(theta)

        polar_lines.append([rho, theta])

    # Convert to numpy array for easier manipulation
    polar_lines = np.array(polar_lines)   
    return polar_lines

# Define custom distance function
# not good, because small theta change will cause big rho difference.
def hough_distance(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2

    # Calculate the difference in rho and theta
    rho_diff = np.abs(rho1 - rho2)
    theta_diff = np.abs(theta1 - theta2)

    # Use some weighting scheme, for example:
    return rho_diff + theta_diff

# use distance between two lines could then be defined as the sum of the 
# distances between the corresponding endpoints. 
def line_distance_intersect_axises(line1, line2):
    # Unpack the endpoints of the lines
    x1_1, y1_1, x2_1, y2_1 = line1
    x1_2, y1_2, x2_2, y2_2 = line2

    # Calculate the intersection points with the x and y axes
    x_int_1 = x1_1 - y1_1 * (x2_1 - x1_1) / (y2_1 - y1_1)
    y_int_1 = y1_1 - x1_1 * (y2_1 - y1_1) / (x2_1 - x1_1)
    x_int_2 = x1_2 - y1_2 * (x2_2 - x1_2) / (y2_2 - y1_2)
    y_int_2 = y1_2 - x1_2 * (y2_2 - y1_2) / (x2_2 - x1_2)

    # Calculate the distances between the corresponding points
    dx = x_int_1 - x_int_2
    dy = y_int_1 - y_int_2

    # Return the sum of the distances
    return np.sqrt(dx**2 + dy**2)

# move the lines together and make them same length before compare
def line_distance(line1, line2):
    # Unpack the endpoints of the lines
    x1_1, y1_1, x2_1, y2_1 = line1
    x1_2, y1_2, x2_2, y2_2 = line2

    # Calculate the lengths of the lines
    length1 = np.sqrt((x2_1 - x1_1)**2 + (y2_1 - y1_1)**2)
    length2 = np.sqrt((x2_2 - x1_2)**2 + (y2_2 - y1_2)**2)

    # Calculate the midpoints of the lines
    mid_x1 = (x1_1 + x2_1) / 2
    mid_y1 = (y1_1 + y2_1) / 2
    mid_x2 = (x1_2 + x2_2) / 2
    mid_y2 = (y1_2 + y2_2) / 2

    # Move the lines to the same x or y coordinate
    if abs(mid_x1 - mid_x2) < abs(mid_y1 - mid_y2):
        # Move to the same x coordinate
        dx = mid_x1 - mid_x2
        x1_2 += dx
        x2_2 += dx
        mid_x2 += dx
    else:
        # Move to the same y coordinate
        dy = mid_y1 - mid_y2
        y1_2 += dy
        y2_2 += dy
        mid_y2 += dy

    # Extend or shrink the second line to the same length as the first line
    scale = length1 / length2
    x1_2 = mid_x2 + (x1_2 - mid_x2) * scale
    y1_2 = mid_y2 + (y1_2 - mid_y2) * scale
    x2_2 = mid_x2 + (x2_2 - mid_x2) * scale
    y2_2 = mid_y2 + (y2_2 - mid_y2) * scale

    # Calculate the distances between the corresponding endpoints
    dx1 = x1_1 - x1_2
    dy1 = y1_1 - y1_2
    dx2 = x2_1 - x2_2
    dy2 = y2_1 - y2_2

    # Return the sum of the distances
    return np.sqrt(dx1**2 + dy1**2) + np.sqrt(dx2**2 + dy2**2)


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
        x1, y1, x2, y2 = line[0]
        # Choose color based on cluster label
        color = colors[labels[i] % len(colors)]
        plt.plot((x1, x2), (y1, y2), color=color)
        # Add text to indicate the cluster number
        if labels[i] != -1:
            plt.text(x1, y1, f'{labels[i]}', color=color)

    # plt's origin is at left bottom, so flip it vertically
    plt.gca().invert_yaxis()
    plt.show()
    
# fine lines nearly paralle
# distance of lines can vary due to the pespective distortion
def nearly_parallel(lines):
    prallel_lines = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            line1 = lines[i]
            line2 = lines[j]
            if abs(line1.orientation - line2.orientation) < orientation_threshold and \
               np.linalg.norm(line1.center - line2.center) < distance_threshold:
                # The lines are nearly parallel and close to each other
                prallel_lines.append(line1, line2)
                

# Homography estimation: OpenCV provides functions to estimate a homography matrix 
# given a set of point correspondences. You would need to manually select four points 
# in your image that form a rectangle, and a corresponding rectangle in a fronto-parallel view
def homo_estimation():
    # Points in the original image
    pts_src = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    # Points in the fronto-parallel view
    pts_dst = np.array([[0, 0], [width, 0], [width, height], [0, height]])

    # Estimate homography
    h, status = cv2.findHomography(pts_src, pts_dst)

    # Warp source image to destination
    im_out = cv2.warpPerspective(im_src, h, (width, height))
        
# Load the image
folder ='/Users/seanmao/Pictures/SEEM/output/Test001'
file = '10_dining table_cropped.png'
image = cv2.imread(os.path.join(folder, file))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

lines = detect_line(gray)
#show_lines(image, lines)
clustering = cluster_lines(lines)
labels = clustering.labels_
show_clusters(lines, labels)
#key_points = ORB(gray)
#show_ORB(image, key_points)
#edges=canny_detection(gray)
#show_canny(image, edges)
#sobelx, sobely, sobelxy = sobel_detection(gray)
#show_sobel_edge(image, sobelx, sobely, sobelxy)
#dst = detect_corner_points(gray)
#show_corners(image, dst)
