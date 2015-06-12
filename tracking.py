import numpy as np

__author__ = 'hikaru'

import cv2
import urllib

hsv_min = np.array([0, 0, 127], np.uint8)
hsv_max = np.array([255, 255, 255], np.uint8)


def HSVChange(x):
    global hsv_min, hsv_max

    hn = cv2.getTrackbarPos('Hn', 'GUI')
    sn = cv2.getTrackbarPos('Sn', 'GUI')
    vn = cv2.getTrackbarPos('Vn', 'GUI')

    hx = cv2.getTrackbarPos('Hx', 'GUI')
    sx = cv2.getTrackbarPos('Sx', 'GUI')
    vx = cv2.getTrackbarPos('Vx', 'GUI')

    hsv_min = np.array([hn, sn, vn], np.uint8)
    hsv_max = np.array([hx, sx, vx], np.uint8)


def initValues():
    global hsv_min, hsv_max
    # Create a black image, a window
    img = np.zeros((128, 256, 3), np.uint8)
    cv2.namedWindow('GUI')

    # create trackbars for color change
    cv2.createTrackbar('Hn', 'GUI', 0, 255, HSVChange)
    cv2.createTrackbar('Sn', 'GUI', 0, 255, HSVChange)
    cv2.createTrackbar('Vn', 'GUI', 0, 255, HSVChange)
    cv2.createTrackbar('Hx', 'GUI', 255, 255, HSVChange)
    cv2.createTrackbar('Sx', 'GUI', 255, 255, HSVChange)
    cv2.createTrackbar('Vx', 'GUI', 255, 255, HSVChange)


def usb():
    global hsv_min, hsv_max

    initValues()
    kernel = np.ones((9, 9), np.uint8)
    alpha = 0.5
    beta = 1.0 - alpha
    gamma = 0.5

    cap = cv2.VideoCapture(0)
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 360)
    while True:
        ret, frame = cap.read()
        if ret:

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # ret, thresh = cv2.threshold(hsv, (h, s, v), 255, cv2.THRESH_BINARY)
            thresh = cv2.inRange(hsv, hsv_min, hsv_max)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            canny = cv2.Canny(opening, 100, 200)
            canny3 = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
            mask_inv = cv2.bitwise_not(canny)

            final = cv2.bitwise_and(canny3, canny3, mask=canny)
            dst = cv2.add(frame, final)

            # cv2.imshow("dst", dst)

            # ret, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            # canny2 = cv2.Canny(thresh2, 100, 200)
            # final2 = cv2.bitwise_and(thresh2, thresh2, mask=canny2)
            # drawing = cv2.add(gray, final2)
            # drawing = cv2.cvtColor(drawing, cv2.COLOR_GRAY2RGB)
            # cv2.imshow("gray", drawing)

            # cv2.imshow("frame", frame)
            # cv2.imshow("hsv", hsv)
            # cv2.imshow("thresh", thresh)
            # cv2.imshow("opening", opening)
            # cv2.imshow("canny", canny)

            # mask = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
            thresh3 = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            vis = np.concatenate((dst, thresh3), axis=0)
            cv2.imshow("vis", vis)

            if cv2.waitKey(1) & 0xff == 27:
                break

    cv2.destroyAllWindows()


def ipcam(url):
    global hsv_min, hsv_max

    initValues()
    stream = urllib.urlopen(url)
    bytes = ''

    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 360)

    '''Variables'''
    kernel = np.ones((9, 9), np.uint8)
    alpha = 0.5
    beta = 1.0 - alpha
    gamma = 0.5
    while True:
        bytes += stream.read(10240)
        a = bytes.find('\xff\xd8')
        b = bytes.find('\xff\xd9')

        ''' decoding the jpg '''
        if a != -1 and b != -1:
            jpg = bytes[a:b + 2]
            bytes = bytes[b + 2:]

            frame = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_COLOR)

            # ret, frame = cap.read()
            # if 1:

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # ret, thresh = cv2.threshold(hsv, (h, s, v), 255, cv2.THRESH_BINARY)
            thresh = cv2.inRange(hsv, hsv_min, hsv_max)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            canny = cv2.Canny(opening, 100, 200)
            canny3 = cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
            mask_inv = cv2.bitwise_not(canny)

            final = cv2.bitwise_and(canny3, canny3, mask=canny)
            dst = cv2.add(frame, final)

            # cv2.imshow("dst", dst)

            # ret, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            # canny2 = cv2.Canny(thresh2, 100, 200)
            # final2 = cv2.bitwise_and(thresh2, thresh2, mask=canny2)
            # drawing = cv2.add(gray, final2)
            # drawing = cv2.cvtColor(drawing, cv2.COLOR_GRAY2RGB)
            # cv2.imshow("gray", drawing)

            # cv2.imshow("frame", frame)
            # cv2.imshow("hsv", hsv)
            # cv2.imshow("thresh", thresh)
            # cv2.imshow("opening", opening)
            # cv2.imshow("canny", canny)

            # mask = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
            thresh3 = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            vis = np.concatenate((dst, thresh3), axis=0)
            cv2.imshow("vis", vis)

            if cv2.waitKey(1) & 0xff == 27:
                break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    usb()
    # ipcam('http://172.20.1.5:8080/?action=stream')
