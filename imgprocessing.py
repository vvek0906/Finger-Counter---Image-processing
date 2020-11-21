import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0) #capturing from camera

while (1):

    try:  # an error comes if it does not find anything in window as it cannot find contour of max area
        # therefore this try error statement

        ret, frame = cap.read() #return parameters of read (success value,image). If success value is correct, image can be used.
        frame = cv2.flip(frame, 1) #flips the frame around the x axis
        kernel = np.ones((3, 3), np.uint8) # returns a matrix of ones with 3*3 size and int type

        # define region of interest
        roi = frame[100:300, 100:300] 

        cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0) #(image,starting point, endpoint,colour , thickness) Rectangle to hold hand in camera.
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) # change roi to HSV from RGB

        # define range of skin color in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # extract skin colur image
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask, kernel, iterations=4) # uses the kernal convolution to dilate the white part more and make it more visible.

        # blur the image 
        mask = cv2.GaussianBlur(mask, (5, 5), 100) #smoothen the image by using gaussian function

        # find contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ''' Parameters
image	: Source, an 8-bit single-channel image. Non-zero pixels are treated as 1's. Zero pixels remain 0's, so the image is treated as binary . You can use compare, inRange, threshold , adaptiveThreshold, Canny, and others to create a binary image out of a grayscale or color one. If mode equals to RETR_CCOMP or RETR_FLOODFILL, the input can also be a 32-bit integer image of labels (CV_32SC1).
contours:	Detected contours. Each contour is stored as a vector of points (e.g. std::vector<std::vector<cv::Point> >).
hierarchy:	Optional output vector (e.g. std::vector<cv::Vec4i>), containing information about the image topology. It has as many elements as the number of contours. For each i-th contour contours[i], the elements hierarchy[i][0] , hierarchy[i][1] , hierarchy[i][2] , and hierarchy[i][3] are set to 0-based indices in contours of the next and previous contours at the same hierarchical level, the first child contour and the parent contour, respectively. If for the contour i there are no next, previous, parent, or nested contours, the corresponding elements of hierarchy[i] will be negative.
mode:	Contour retrieval mode, see RetrievalModes
method:	Contour approximation method, see ContourApproximationModes
offset:	Optional offset by which every contour point is shifted. This is useful if the contours are extracted from the image ROI and then they should be analyzed in the whole image context.'''
        
        # find contour of max area(hand)
        cnt = max(contours, key=lambda x: cv2.contourArea(x))

        # approx the contour a little
        epsilon = 0.0005 * cv2.arcLength(cnt, True) # adjust the arc length by a factor
        approx = cv2.approxPolyDP(cnt, epsilon, True) # make the polygram more curved in nature

        # make convex hull around hand. A convex hull is a curved surface that encompasses the area
        hull = cv2.convexHull(cnt) 

        # define area of hull and area of hand
        areahull = cv2.contourArea(hull) 
        areacnt = cv2.contourArea(cnt)

        # find the percentage of area not covered by hand in convex hull
        arearatio = ((areahull - areacnt) / areacnt) * 100

        # find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull) #Any deviation of the object from this hull can be considered as convexity defect.

        # l = no. of defects
        l = 0

        # code for finding no. of defects due to fingers
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt = (100, 180)

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            s = (a + b + c) / 2
            ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

            # distance between point and convex hull
            d = (2 * ar) / a

            # apply cosine rule here
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d > 30:
                l += 1
                cv2.circle(roi, far, 3, [255, 0, 0], -1)

            # draw lines around hand
            cv2.line(roi, start, end, [0, 255, 0], 2)

        l += 1

        # print corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l == 1:
            if areacnt < 2000: # condition represents closed fist not presented properly.
                cv2.putText(frame, 'Put hand in the box', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                if arearatio < 12:
                    cv2.putText(frame, '0', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                elif arearatio < 17.5:
                    cv2.putText(frame, 'Best of luck', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

                else:
                    cv2.putText(frame, '1', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 2:
            cv2.putText(frame, '2', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 3:

            if arearatio < 27:
                cv2.putText(frame, '3', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, 'ok', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 4:
            cv2.putText(frame, '4', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 5:
            cv2.putText(frame, '5', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 6:
            cv2.putText(frame, 'reposition', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        else:
            cv2.putText(frame, 'reposition', (10, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        # show the windows
        cv2.imshow('mask', mask)
        cv2.imshow('frame', frame)
    except:
        pass

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
