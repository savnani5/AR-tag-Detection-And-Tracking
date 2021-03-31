import numpy as np
from One_b import detect_orientation_and_id 
import cv2

def svd(X):
    """
    Let X be an m*n matrix, so to find SVD we have 3 conditions depending on the values of m and n:
    
    1) m = n : Here the X matrix is a square matrix i.e the number of coefficients is equal to the number of data points. Also, here the sigma matrix will be square so it is simple to calculate.
    2) m < n : Here the X matrix is a rectangular matrix i.e number of coefficients is more than the number of data points, thus this can't be the case for fitting the parabola, but will be used in homography.
    3) m > n : Here the X matrix is a rectangular matrix i.e number of coefficients is less then the number of data points (overdefined situation), and we'll be dealing with this case in total least squares.    
    """

    #_________________ Calculating Vt matrix________________________

    r_values, r_vectors = np.linalg.eig(np.dot(X.T,X))

    # Sorting the values and vectors in descending order
    idx = r_values.argsort()[::-1]   
    r_values = r_values[idx]
    r_vectors = r_vectors[:,idx]
    Vt = r_vectors.T
    V = Vt.T
    # print("vt", Vt)
    
    #________________Calculating Sigma matrix_______________________

    # Removing "zero valued" eigen values
    index = []
    for i in range(len(r_values)):
        if r_values[i] <= 0.001:
            index.append(i)

    r_values = np.delete(r_values, index)

    s = np.zeros(shape=(X.shape))
    for i in range(len(r_values)):
        s[i,i] = r_values[i]**0.5
    # print("s", s)

    #__________________Calculating U matrix____________________________

    l_values, l_vectors = np.linalg.eig(np.dot(X,X.T))
    idx = l_values.argsort()[::-1]   
    l_values = l_values[idx]
    l_vectors = l_vectors[:,idx]
    U = l_vectors.real
    # print("U",U)
    
    return U, s, Vt

def get_homography(sp, dp):
    """ To compute the homography matrix we have to solve the homogeneous equation AH = 0, where H is the homography matrix"""

    # Constructing the A matrix
    A = np.zeros(shape=(8,9))
    c = 0
    for i in range(0,len(2*sp),2):
        A[i] = np.array([-sp[c][0], -sp[c][1], -1, 0, 0, 0, sp[c][0]*dp[c][0], sp[c][1]*dp[c][0], dp[c][0]])
        A[i+1] = np.array([0, 0, 0, -sp[c][0], -sp[c][1], -1, sp[c][0]*dp[c][1], sp[c][1]*dp[c][1], dp[c][1]]) 
        c+=1

    # Using SVD to find the H matrix
    U, s, Vt = svd(A)
    H = Vt[-1,:]
    H = H.reshape(3,3)
    return H

def warp_perspective(src_img, dst_img, H):
    """This function is responsible for warping the image using the homography matrix."""
    print("src image", src_img.shape)
    for i in range(src_img.shape[1]):
        for j in range(src_img.shape[0]):
            src_cood = np.array([i,j,1])
            result = np.dot(H, src_cood)
            try:
                u = int(result[0]/result[2])
                v = int(result[1]/result[2])
                # This is in try statement to avoid getting the index error due to mapping of points in source image outside the bounds of destination image
                dst_img[v,u] = src_img[j,i]
            except:
                pass
    return dst_img

def orientation_detection(src_img, src_points):
    """This function maps the coordinates of the ar tag and detects the orientation and id of the ar tag.
    IMP_Note: The ar tag is mapped to a 32*32 region to prevent holes in mapping and then for detection it 
    is resized using linear interpolation."""
    
    dst_img = np.zeros(shape=src_img.shape, dtype=np.uint8)
    dst_pts_for_ar_tag = [(0,0), (0,32), (32,32), (32,0)]
    H = get_homography(src_points, dst_pts_for_ar_tag)
    warped_image = warp_perspective(src_img, dst_img, H)
    ar_tag = warped_image[:32, :32]
    ar_tag = cv2.resize(ar_tag, (64,64), interpolation = cv2.INTER_AREA)
    return detect_orientation_and_id(ar_tag)

 