import cv2

trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

img = cv2.imread('RDJ.png')

grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Elvos Face Detector', grayscaled_img)
cv2.waitKey()


print("code complete")
