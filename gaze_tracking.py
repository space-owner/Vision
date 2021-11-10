from detector.face_detector import MTCNNFaceDetector
from models.elg_keras import KerasELG
from keras import backend as K
import numpy as np
import cv2
from matplotlib import pyplot as plt

def draw_pupil(im, inp_im, lms):
    draw = im.copy()
    draw = cv2.resize(draw, (inp_im.shape[2], inp_im.shape[1]))
    pupil_center = np.zeros((2,))
    pnts_outerline = []
    pnts_innerline = []
    # stroke = inp_im.shape[1] // 12 + 1
    stroke = 4
    for i, lm in enumerate(np.squeeze(lms)):
        #print(lm)
        y, x = int(lm[0]*3), int(lm[1]*3)

        if i < 8:
            draw = cv2.circle(draw, (y, x), stroke, (125,255,125), -1)
            pnts_outerline.append([y, x])
        elif i < 16:
            draw = cv2.circle(draw, (y, x), stroke, (125,125,255), -1)
            pnts_innerline.append([y, x])
            pupil_center += (y,x)
        elif i < 17:
            draw = cv2.drawMarker(draw, (y, x), (255,200,200), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=stroke, line_type=cv2.LINE_AA)
        else:
            draw = cv2.drawMarker(draw, (y, x), (255,125,125), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=stroke, line_type=cv2.LINE_AA)
    pupil_center = (pupil_center/8).astype(np.int32)
    draw = cv2.circle(draw, (pupil_center[0], pupil_center[1]), stroke, (255,255,0), -1)
    draw = cv2.polylines(draw, [np.array(pnts_outerline).reshape(-1,1,2)], isClosed=True, color=(125,255,125), thickness=stroke//2)
    draw = cv2.polylines(draw, [np.array(pnts_innerline).reshape(-1,1,2)], isClosed=True, color=(125,125,255), thickness=stroke//2)
    return draw


mtcnn_weights_dir = "./mtcnn_weights/"
fd = MTCNNFaceDetector(sess=K.get_session(), model_path=mtcnn_weights_dir)

model = KerasELG()
model.net.load_weights("./elg_weights/elg_keras.h5")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024*2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768*2)

while True:
    ret, input_img = cap.read()

    # assuming there is only one face in input image
    face, lms = fd.detect_face(input_img)

    if len(face) > 1:
        print("another face detected.")
        cv2.waitKey(100)
        continue
    try:
        cv2.rectangle(input_img, (int(face[0][1]), int(face[0][0])), (int(face[0][3]), int(face[0][2])), (0, 255, 0), 2)
    except IndexError:
        print("no face detected")
        continue

    # print("face detected")

    # eyes detecting...
    left_eye_xy = np.array([lms[6], lms[1]])
    right_eye_xy = np.array([lms[5], lms[0]])

    dist_eyes = np.linalg.norm(left_eye_xy - right_eye_xy)
    eye_bbox_w = (dist_eyes / 1.25)
    eye_bbox_h = (eye_bbox_w *0.6)

    left_eye_im = input_img[
        int(left_eye_xy[0]-eye_bbox_h//2):int(left_eye_xy[0]+eye_bbox_h//2),
        int(left_eye_xy[1]-eye_bbox_w//2):int(left_eye_xy[1]+eye_bbox_w//2), :]
    # left_eye_im = left_eye_im[:,::-1,:] # No need for flipping left eye for iris detection
    right_eye_im = input_img[
        int(right_eye_xy[0]-eye_bbox_h//2):int(right_eye_xy[0]+eye_bbox_h//2),
        int(right_eye_xy[1]-eye_bbox_w//2):int(right_eye_xy[1]+eye_bbox_w//2), :]

    # prepare a image.
    inp_left = cv2.cvtColor(left_eye_im, cv2.COLOR_RGB2GRAY)
    inp_left = cv2.equalizeHist(inp_left)
    inp_left = cv2.resize(inp_left, (180,108))[np.newaxis, ..., np.newaxis]

    inp_right = cv2.cvtColor(right_eye_im, cv2.COLOR_RGB2GRAY)
    inp_right = cv2.equalizeHist(inp_right)
    inp_right = cv2.resize(inp_right, (180,108))[np.newaxis, ..., np.newaxis]

    # Predict eye region landmarks.
    input_array = np.concatenate([inp_left, inp_right], axis=0)
    pred_left, pred_right = model.net.predict(input_array/255 * 2 - 1)

    lms_left = model._calculate_landmarks(pred_left)
    result_left = draw_pupil(left_eye_im, inp_left, lms_left)
    lms_right = model._calculate_landmarks(pred_right)
    result_right = draw_pupil(right_eye_im, inp_right, lms_right)

    # draw the eyes...
    # draw = input_img.copy()

    slice_h = slice(int(left_eye_xy[0]-eye_bbox_h//2), int(left_eye_xy[0]+eye_bbox_h//2))
    slice_w = slice(int(left_eye_xy[1]-eye_bbox_w//2), int(left_eye_xy[1]+eye_bbox_w//2))
    im_shape = left_eye_im.shape[::-1]

    input_img[slice_h, slice_w, :] = cv2.resize(result_left, im_shape[1:])

    slice_h = slice(int(right_eye_xy[0]-eye_bbox_h//2), int(right_eye_xy[0]+eye_bbox_h//2))
    slice_w = slice(int(right_eye_xy[1]-eye_bbox_w//2), int(right_eye_xy[1]+eye_bbox_w//2))
    im_shape = right_eye_im.shape[::-1]

    input_img[slice_h, slice_w, :] = cv2.resize(result_right, im_shape[1:])

    cv2.imshow("result", input_img)
    cv2.waitKey(100)
    # TODO : destory all ...?