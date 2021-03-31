import cv2
import numpy as np
from matplotlib import pyplot as plt

from One_b import detect_orientation_and_id 
from utils import get_homography, warp_perspective, orientation_detection


print("Please enter the Number to run the code for the Corresponding video:\n 1. Tag0.mp4\n 2. Tag1.mp4\n 3. Tag2.mp4\n 4. multipleTags.mp4\n")
video_list = ['Tag0.mp4', 'Tag1.mp4', 'Tag2.mp4', 'multipleTags.mp4']
n = int(input())
video = video_list[n-1]

cap = cv2.VideoCapture(video)
frame_width = int(cap.get(3)*0.7)
frame_height = int(cap.get(4)*0.7)

frametime = 1 # To control the video playback speed
result = cv2.VideoWriter(f'{video.strip(".mp4")}_output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width, frame_height)) 

testudo = cv2.imread("testudo.png")
testudo = cv2.resize(testudo, (100,100), interpolation = cv2.INTER_AREA)
gray_testudo = cv2.cvtColor(testudo, cv2.COLOR_BGR2GRAY)

try:
    while(cap.isOpened()):
        
        _, frame = cap.read()
        width = int(frame.shape[1] * 0.7)
        height = int(frame.shape[0] * 0.7)

        frame = cv2.resize(frame, (width, height), interpolation = cv2.INTER_AREA)
        frame1 = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(gray, 210, 255,cv2.THRESH_BINARY_INV)
        print("frameshape",frame.shape)

        #_____________________________Noise Eradication_________________________________
        
        if video != 'multipleTags.mp4':
            kernel = np.ones((5,5),np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            # External Noise Removal based on ROI
            opening[:, frame.shape[1] - int((frame.shape[0]*0.2)):frame.shape[1]] = 255
            opening[0:int((frame.shape[0]*0.2)), :] = 255
        else:
            opening = thresh
        # blur = cv2.medianBlur(img_back, 3)
        # blur = cv2.GaussianBlur(img_back,(3,3),0)
        # _____________________________________________________________________________
        
        #__________________Corner detection using Contours___________________________________________
        
        contours, hierarchy = cv2.findContours(opening,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # frame = cv2.drawContours(frame, contours, -1, (0,255,0), 2)

        # hierarchy of each contour: [Next, Previous, First_Child, Parent]

        print("hierarchy", hierarchy)
        child_list = []
        for h in hierarchy[0]:
            if h[2] != -1:
                child_list.append(h[2])
        
        print(child_list)

        sec_child_list = []
        if len(child_list) == 0:
            sec_child_list.append(2)
        elif len(child_list) == 2:
            # Its a 3 level contour (for cases when the white page is cut from video frame)
            sec_child_list.append(child_list[1] - 2)  
        elif len(child_list) > 2 and len(child_list) <= 4:
            sec_child_list.append(child_list[1])
            # Its a 4/5 level contour
        elif len(child_list) > 4:
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
        print(sec_child_list)

        for sec_child in sec_child_list: 
        
            cnt = contours[sec_child]
            approx = cv2.approxPolyDP(cnt, 0.1099* cv2.arcLength(cnt, True), True)
            print(approx)

            # Condition removing frame boundary as contour
            if approx[0][0][0] == 0 and approx[0][0][1] == 0:
                sec_child = 1
                cnt = contours[sec_child]
                approx = cv2.approxPolyDP(cnt, 0.1099* cv2.arcLength(cnt, True), True)
                print(approx)

            dst_points = []
            for point in approx:
                dst_points.append(tuple(point[0]))
                # cv2.circle(frame, tuple(point[0]), 3, (255,0,0), 3)

            source_points = [(0,0), (0,testudo.shape[1]), (testudo.shape[1],testudo.shape[0]), (testudo.shape[0],0)]
            
            # Logic to get min and max points to get a bounding box to perform homography
            unzipped_values = zip(*dst_points)
            unzipped_list = list(unzipped_values)
            min_x = min(unzipped_list[0])
            max_x = max(unzipped_list[0])
            min_y = min(unzipped_list[1])
            max_y = max(unzipped_list[1])

            d_frame = frame[min_y:max_y, min_x:max_x].copy()

            # Calculating new dst points wrt new coordinates 
            n_dst_points  = [(point[0]-min_x, point[1]-min_y) for point in dst_points]

            if len(dst_points) == 4:
                
                # Orientation detection 
                orientation, tag_id, tag_thresh, actions = orientation_detection(d_frame, n_dst_points)

                ## Orientation correction
                for i in range(len(actions)):
                    source_points.append(source_points.pop(0))
            
                H = get_homography(source_points, dst_points)
                # print("H mat", H)
                warped_testudo = warp_perspective(testudo, frame, H)
        
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

