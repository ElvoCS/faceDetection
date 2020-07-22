import cv2

# load pre trained data on face frontals
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# choose an img to detect faces in
#img = cv2.imread('groupPhoto.png')

# capture video from webcam
webcam = cv2.VideoCapture(0)
key = cv2.waitKey(1)


# iterate over frames
while True:
    # read current frame
    succesful_frame_read, frame = webcam.read()

    # must convert to greyscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # draw rectangles around faces, goes through list staring at 0 and squares each face  it sees
    for(x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Elvos Face Detector', frame)
    key = cv2.waitKey(1)

    # press q to quit
    if key == 81 or key == 113:
        break


"""
# detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# draw rectangles around faces, goes through list staring at 0 and squares each face  it sees
for(x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# print(face_coordinates)

# show img
cv2.imshow('Elvos Face Detector', img)
cv2.waitKey()
"""
print("code complete")
