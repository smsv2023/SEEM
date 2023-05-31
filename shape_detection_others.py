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
