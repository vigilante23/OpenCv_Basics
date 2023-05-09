import cv2
import mediapipe as mp

def process_img(image, face_detection):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1,y1,w,h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1 = int(x1*W)
            y1 = int(y1*H)
            w = int(w*W)
            h = int(h*H)
            image[y1:y1 +h, x1:x1 + w, :] = cv2.blur(image[y1:y1 +h, x1:x1 + w, :], (40,40))
    return image

    
image = cv2.imread('testImg.png')
H,W, _= image.shape

mp_face_detection = mp.solutions.face_detection
with mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence = 0.5) as face_detection:
    image = process_img(image, face_detection)
    
    

cv2.imwrite('.\img.jpg', image)