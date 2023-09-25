import cv2
import time
# inspiration: https://www.datacamp.com/tutorial/face-detection-python-opencv
# inspiration: https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/

image_name = "test1.jpg"
video_name = "test1.mp4"


def find_face(im_name):
    # load image
    image = cv2.imread(image_name)
    # convert to greyscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # load built-in classifier
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades
                                       + "haarcascade_frontalface_default.xml")
    # detection
    detected = classifier.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=8, minSize=(40, 40))
    # drawing rectangle
    for(x, y, w, h) in detected:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 4)
    # saving image
    cv2.imwrite("marked_face.jpg", image)


def find_face_vid(image):
    # convert to greyscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # load built-in classifier
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades
                                       + "haarcascade_frontalface_default.xml")
    # detection
    detected = classifier.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=8, minSize=(40, 40))
    # drawing rectangle
    for(x, y, w, h) in detected:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 4)
    # returning modified image
    return image


def find_face_video(vid_name):
    # open video
    video = cv2.VideoCapture(vid_name)
    # read resolution
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))

    # read video frame by frame
    play, frame = video.read()
    # write video to file
    out_video = cv2.VideoWriter('output.avi', cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
    while play:
        out_video.write(find_face_vid(frame))
        play, frame = video.read()

    # close video
    video.release()
    out_video.release()


if __name__ == '__main__':
    find_face(image_name)
    print("Starting process, finding faces in: " + video_name)
    find_face_video(video_name)
    print("Process ended, saved to: " + "output.avi")
    cv2.destroyAllWindows()


