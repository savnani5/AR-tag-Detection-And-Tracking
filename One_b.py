import cv2
import time
import numpy as np
from matplotlib import pyplot as plt

def calculate_mean(image, start, side):
    # This function calculates the mean for the given ROI of the image
    roi = image[start[0]:(start[0]+side), start[1]:(start[1]+side)]
    sum = 0
    for i in range(roi.shape[0]):
        for j in range(roi.shape[1]):
            sum += roi[i][j]
        try:
            mean = sum/(side**2)
        except:
            mean =sum/(8**2) 
    return mean


def detect_orientation_and_id(ar_tag):
    # This function detects the orientation ,ID and actions required to correc the superimposition.
    rows, cols, _ = ar_tag.shape
    gray = cv2.cvtColor(ar_tag, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)

    thresh[0:int(rows/4),:] = 0
    thresh[int(3*rows/4):int(rows),:] = 0
    thresh[:,0:int(cols/4)] = 0
    thresh[:,int(3*cols/4):int(cols)] = 0

    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.erode(thresh,kernel,iterations = 1)

    # Detecting corners is unreliable for video frame images thats why checking avergae is used, it is more robust !

    # corners = cv2.goodFeaturesToTrack(thresh, 25, 0.1, 10)
    # corners = np.int0(corners)

    # for i in corners:
    #     x,y = i.ravel()
    #     print(x,y)
    #     cv2.circle(ar_tag,(x,y), 3 ,(255,0,255), -1)

    mean_outer_ul = calculate_mean(thresh, ((int(2*thresh.shape[0]/8), int(2*thresh.shape[0]/8))), int(thresh.shape[0]/8))
    mean_outer_ur = calculate_mean(thresh, ((int(2*thresh.shape[0]/8), int(5*thresh.shape[0]/8))), int(thresh.shape[0]/8))
    mean_outer_dr = calculate_mean(thresh, ((int(5*thresh.shape[0]/8), int(5*thresh.shape[0]/8))), int(thresh.shape[0]/8))
    mean_outer_dl = calculate_mean(thresh, ((int(5*thresh.shape[0]/8), int(2*thresh.shape[0]/8))), int(thresh.shape[0]/8))

    mean_inner_ul = calculate_mean(thresh, ((int(3*thresh.shape[0]/8), int(3*thresh.shape[0]/8))), int(thresh.shape[0]/8))
    mean_inner_ur = calculate_mean(thresh, ((int(3*thresh.shape[0]/8), int(4*thresh.shape[0]/8))), int(thresh.shape[0]/8))
    mean_inner_dr = calculate_mean(thresh, ((int(4*thresh.shape[0]/8), int(4*thresh.shape[0]/8))), int(thresh.shape[0]/8))
    mean_inner_dl = calculate_mean(thresh, ((int(4*thresh.shape[0]/8), int(3*thresh.shape[0]/8))), int(thresh.shape[0]/8))

    mean_outer_list = [mean_outer_dl, mean_outer_dr, mean_outer_ur, mean_outer_ul]
    mean_inner_list = [mean_inner_dl, mean_inner_dr, mean_inner_ur, mean_inner_ul]
    

    orientation = ['0' if mean < 127.5 else '1' for mean in mean_outer_list]
    tag_id = ['0' if mean < 127.5 else '1' for mean in mean_inner_list]

    org_orientation = orientation.copy()

    # Correction for Upright position
    actions = []
    # Right shifting the digits
    for i in range(3):
        if orientation == ['0','1','0','0']:
            break
        orientation.insert(0, orientation.pop())
        actions.append('AC')
        tag_id.insert(0,tag_id.pop())

    return org_orientation, tag_id, thresh, actions

if __name__=="__main__":

    ar_tag = cv2.imread("ref_marker.png")
    orientation, tag_id, thresh, _ = detect_orientation_and_id(ar_tag)
    print("Current orientation(Anticlockwise Order- Bottom Left to Top left): ", ''.join(orientation))
    print("\nID of the AR tag with respect to the upright orientation(Anticlockwise Order): ",''.join(tag_id))
    cv2.imshow("Ar_Tag", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
