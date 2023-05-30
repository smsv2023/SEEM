import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

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

# Load the image
folder ='/Users/seanmao/Pictures/SEEM/output/Test001'
file = '10_dining table_cropped.png'
image = cv2.imread(os.path.join(folder, file))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

key_points = ORB(gray)
show_ORB(image, key_points)
#edges=canny_detection(gray)
#show_canny(image, edges)
#sobelx, sobely, sobelxy = sobel_detection(gray)
#show_sobel_edge(image, sobelx, sobely, sobelxy)
#lines = detect_line(gray)
#show_lines(image, lines)
#dst = detect_corner_points(gray)
#show_corners(image, dst)
