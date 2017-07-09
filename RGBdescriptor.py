import numpy as np 
import cv2

# bins are basically the X-axis and the values are ploted on y-axis in a histogram.


class HistogramBGR:
    def __init__(self,bins):
        # bins that histogram will use(int)
        self.bins = bins

    def featurize(self,image):
        #                   image,    channels, mask, histSize,   rangesOfEachChannel
        hist = cv2.calcHist([image], [0, 1, 2], None, self.bins, [0, 256, 0, 256, 0, 256])
        # normalizing in terms of pixel count say 20% insted of 120 pixel
        hist = cv2.normalize(hist, hist)   

        #finally Flatten bin*bin*bin
        return hist.flatten()
        

