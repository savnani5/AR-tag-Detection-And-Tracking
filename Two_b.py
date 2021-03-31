import cv2
import numpy as np
from matplotlib import pyplot as plt

from Kalman_filter import KalmanFilter  
from One_b import detect_orientation_and_id 
from utils import get_homography, warp_perspective

# We cannot choose the parameters of the Kalman filter as the acceleration in both directions and std. deviations are not known. 
# Here, the parameters are just assumed, to apply the filter.
KF = KalmanFilter(0.1, 2, 2, 3, 0.2,0.2)


print("Please enter the Number to run the code for the Corresponding video:\n 1. Tag0.mp4\n 2. Tag1.mp4\n 3. Tag2.mp4\n 4. multipleTags.mp4\n")
video_list = ['Tag0.mp4', 'Tag1.mp4', 'Tag2.mp4', 'multipleTags.mp4']
n = int(input())
video = video_list[n-1]


K_matrix = np.array([[1406.08415449821, 2.20679787308599, 1014.13643417416], 
                     [        0,        1417.99930662800, 566.347754321696], 
                     [        0,               0,              1          ]])


cap = cv2.VideoCapture(video)
frame_width = int(cap.get(3)*0.7)
frame_height = int(cap.get(4)*0.7)
frametime = 100 # To control the video playback speed
result = cv2.VideoWriter(f'{video.strip(".mp4")}_cube_output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width, frame_height)) 

testudo = cv2.imread("testudo.png")
testudo = cv2.resize(testudo, (128,128), interpolation = cv2.INTER_AREA)
gray_testudo = cv2.cvtColor(testudo, cv2.COLOR_BGR2GRAY)

try:
    while(cap.isOpened()):
        
        _, frame = cap.read()
        width = int(frame.shape[1] * 0.7)
        height = int(frame.shape[0] * 0.7)

        frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray, 210, 255,cv2.THRESH_BINARY_INV)
        print("frameshape",frame.shape)

        #______________________Noise Eradication____________________________________
        if video != 'multipleTags.mp4':
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            # External Noise Removal based on ROI
            opening[:, frame.shape[1] - int((frame.shape[0]*0.2)):frame.shape[1]] = 255
            opening[0:int((frame.shape[0]*0.2)), :] = 255
        else:
            opening = thresh
        
        # blur = cv2.medianBlur(img_back, 3)
        # blur = cv2.GaussianBlur(img_back,(3,3),0)
        # _____________________________________________________________________________
        
        #__________________Corner detection using Contours________________________________________________
        
        contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # frame = cv2.drawContours(frame, contours, -1, (0,255,0), 2)

        # hierarchy of each contour: [Next, Previous, First_Child, Parent]

        print("hierarchy", hierarchy)
        child_list = []
        for h in hierarchy[0]:
            if h[2] != -1:
                child_list.append(h[2])
        
        sec_child_list = []
        if len(child_list) == 0:
            sec_child_list.append(2)
        elif len(child_list) == 2:
            # Its a 3 level contour (for cases when the white page is cut from video frame)
            sec_child_list.append(child_list[1] - 2)  
        elif len(child_list) > 2 and len(child_list) < 4:
            sec_child_list.append(child_list[1])
            # Its a 4/5 level contour
        elif len(child_list) >= 4:
            # Contours in multi tag video Accomodate 2 cases white paper cut and normal ones
            if 2 in child_list and 3 in child_list: 
                for i in range(len(child_list)):
                    if (child_list[i] - child_list[i-1]) != 1:
                        if child_list[i] == 1:
                            sec_child_list.append(child_list[i+1])
                        else:
                            sec_child_list.append(child_list[i])
            else:     
                for i in range(len(child_list)):
                    if (child_list[i] - child_list[i-1]) != 1:
                        if child_list[i] == 1:
                            sec_child_list.append(child_list[i]-1)
                        elif child_list[i] == 3:
                            sec_child_list.append(child_list[i]+1)
                        elif child_list[i] == 4:
                            sec_child_list.append(child_list[i]+1)
                        else:
                            sec_child_list.append(child_list[i])
        else:
            sec_child_list.append(child_list[0]-1)
            
        for sec_child in sec_child_list: 
            cnt = contours[sec_child]
            approx = cv2.approxPolyDP(cnt, 0.1099* cv2.arcLength(cnt, True), True)
            
            dst_points = []
            for point in approx:
                dst_points.append(tuple(point[0]))
                # cv2.circle(frame, tuple(point[0]), 3, (255,0,0), 3)

            source_points = [(0,testudo.shape[0]), (0,0), (testudo.shape[1],0), (testudo.shape[1],testudo.shape[0])]
            
            if len(dst_points)== 4: 
            
                #_________________________Calculating Projection matrix__________________________
                H = get_homography(source_points, dst_points)    
                H = H.T
                h1, h2, h3 = H[0], H[1], H[2]
                K_inv = np.linalg.inv(K_matrix)
                L = 2 / (np.linalg.norm(np.dot(K_inv, h1)) + np.linalg.norm(np.dot(K_inv, h2)))
                r1 = L * np.dot(K_inv, h1)
                r2 = L * np.dot(K_inv, h2)
                r3 = np.cross(r1, r2)
                T = L * (np.dot(K_inv, h3.reshape(3, 1)))
                R = np.array([[r1], [r2], [r3]])
                R = np.reshape(R, (3, 3))
                R_t = np.append(R, T, axis=1).reshape(3,4)
                P = np.dot(K_matrix, R_t)


                points = [[point[0], point[1], -60, 1] for point in source_points]
                z_points = []
                for point in points:
                    a = np.array(point)
                    x = np.dot(P, a)
                    print(x)
                    x[0] = x[0]/x[2]
                    x[1] = x[1]/x[2]
                    z_points.append((x[0], x[1]))
                    # cv2.circle(frame, (int(x[0]), int(x[1])), 3, (255,0,0), 3)

                #______________________________Kalman Filter_______________________________
                predicted_points = []
                for i in range(4):
                    (x, y) = KF.predict()
                    predicted_points.append((x[0,0], y[0,0]))
                
                updated_points = []
                for point in z_points:
                    (x1, y1) = KF.update((point[0], point[1]))
                    print("x1,y1", (x1,y1))
                    updated_points.append((x1[0,0], y1[0,1]))
                # print("predicted points",predicted_points)
                # print("updated points", updated_points)
                # ____________________________________________________________________________ 
                # Drawing the cube edges
                for i in range(len(dst_points)):
                    cv2.line(frame, (dst_points[i][0], dst_points[i][1]), (int(z_points[i][0]), int(z_points[i][1])), (255,0,0), 4)
                    if i != len(points)-1:
                        cv2.line(frame, (dst_points[i][0], dst_points[i][1]), (dst_points[i+1][0], dst_points[i+1][1]), (0,0,255), 4)
                        cv2.line(frame, (int(z_points[i][0]), int(z_points[i][1])), (int(z_points[i+1][0]), int(z_points[i+1][1])), (0,0,255), 4)
                        cv2.line(frame, (int(updated_points[i][0]), int(updated_points[i][1])), (int(updated_points[i+1][0]), int(updated_points[i+1][1])), (255,0,255), 4)
                    else:
                        cv2.line(frame, (dst_points[i][0], dst_points[i][1]), (dst_points[0][0], dst_points[0][1]), (0,0,255), 4)
                        cv2.line(frame, (int(z_points[i][0]), int(z_points[i][1])), (int(z_points[0][0]), int(z_points[0][1])), (0,0,255), 4)
                        cv2.line(frame, (int(updated_points[i][0]), int(updated_points[i][1])), (int(updated_points[0][0]), int(updated_points[0][1])), (255,0,255), 4)
                        
                # _____________________________________________________________________________________

        cv2.imshow('frame', frame)
        # cv2.imshow('thresh', thresh)   
        # cv2.imshow('opening', opening)
        result.write(frame)  

        k = cv2.waitKey(frametime) & 0xFF
        if k == 27:
            break
except:
    pass


cap.release()
result.release()
cv2.destroyAllWindows()

