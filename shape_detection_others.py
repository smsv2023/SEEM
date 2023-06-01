import cv2
import numpy as np
import matplotlib.pyplot as plt

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
        x1, y1, x2, y2 = line

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

# define decorator to be used by DBSCAN clustering
def line_distance_decorator(width, height):
    def line_distance_decorated(line1, line2):
        return line_distance(line1, line2, width, height)
    return line_distance_decorated

