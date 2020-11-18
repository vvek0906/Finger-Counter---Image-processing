# Finger-Counter---Image-processing
Project is based on image processing concepts using python and openCV.

Frames from camera feed are captured. Each frame is then processed with pixel pruning and filtering from RGB to greyscale. Frame is blurred to smoothen the boundaries and scaled with respect to square placeholder on frame. Square placeholder for hand positioning is fixed and doesn't move.           
Post Processing, countours formed by hand are retrieved. Contours are sorted and contour with maximum area is rejected. Among rest of the contours, their centers are determmined. Center making an angle more than 15 degree are considered and contour containing such centers are considered. Based on number of such valid contours, fingers count is accepted.
