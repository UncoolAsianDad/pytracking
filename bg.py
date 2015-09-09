import numpy as np
import cv2
import urllib


#cap = cv2.VideoCapture('vtest.avi')
cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

#fgbg = cv2.createBackgroundSubtractorMOG()
fgbg = cv2.BackgroundSubtractorMOG()
url = 'http://camera1/video/mjpg.cgi'
stream = urllib.urlopen(url)
bytes = ''

# cap = cv2.VideoCapture(0)
# cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 360)

'''Variables'''
while True:
    bytes += stream.read(10240)
    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')

    ''' decoding the jpg '''
    if a != -1 and b != -1:
        jpg = bytes[a:b + 2]
        bytes = bytes[b + 2:]

        frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
        
        fgmask = fgbg.apply(frame)        
        cv2.imshow('frame',fgmask)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

