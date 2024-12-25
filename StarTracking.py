import cv2
import numpy as np

#cam = cv2.VideoCapture('./test_video.mp4')
cam = cv2.VideoCapture(0)

class Tracker:
    def __init__(self, x, y):
        self.x = [x]
        self.y = [y]


class CameraFeed:
    def __init__(self):

        self.max_trackers = 10
        self.trackers = []

    def run(self):
        global cam

        while cam.isOpened():
            ret, frame = self.update_frame()

            if ret:
                self.trackers = self.add_trackers(image=frame)

                if self.trackers:

                    for tracker in self.trackers:
                        x, y, w, h = tracker['pos']
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

                    cv2.imshow('Live', frame)
                    cv2.waitKey(1)

            #else:
                #cam = cv2.VideoCapture('./test_video.mp4')
    
    def update_frame(self):
        return cam.read()

    def add_trackers(self, image):
        
        trackers = []

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 225])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        frame = cv2.bitwise_and(image, image, mask=mask)
        contours = self.get_contours(image=frame)

        min_size = 15

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w <= min_size and h <= min_size:
                cropped_image = frame[y:y+h, x:x+w]
                gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                mean = np.mean(gray)
                max_bright = {'max': mean, 'pos': (x, y, w, h)}
                trackers.append(max_bright)


        for i in range(len(trackers)):
            for j in range(0, len(trackers)-i-1):
                if trackers[j]['max'] > trackers[j+1]['max']:
                    trackers[j]['max'], trackers[j+1]['max'] = trackers[j+1]['max'], trackers[j]['max']

        cv2.imshow('Mask', frame)
        return trackers

    
    def get_contours(self, image):
        edges = cv2.Canny(image, 255, 255)
        cv2.imshow('Edges', edges)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours

if __name__ == '__main__':

    app = CameraFeed()
    app.run()

print('Ended')
