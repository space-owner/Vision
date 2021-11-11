from detector.face_detector import MTCNNFaceDetector
from models.elg_keras import KerasELG
from keras import backend as K
import numpy as np
import cv2

def draw_pupil(im, inp_im, lms):
    draw = im.copy()
    draw = cv2.resize(draw, (inp_im.shape[2], inp_im.shape[1]))
    eye_lm = [[int(x*3), int(y*3)] for x, y in np.squeeze(lms)]

    # eyeball_centre = np.mean(eye_lm[:8], axis=0).astype(np.int32)
    eyeball_centre = np.array(eye_lm[17]).astype(np.int32)

    # pupil_center = np.mean(eye_lm[8:16], axis=0).astype(np.int32)
    pupil_center = np.array(eye_lm[16]).astype(np.int32)

    # TODO: eye radius needs a weights parameter for estimation.
    eye_radius = np.linalg.norm(eye_lm[0] - eyeball_centre) * 5
    print("eye_radius", eye_radius)
    draw = cv2.polylines(draw, [np.array(eye_lm[8:16]).reshape(-1, 1, 2)], isClosed=True, color=(125, 125, 255), thickness=1)
    return draw, eye_lm, eyeball_centre, pupil_center, eye_radius

def draw_gaze(image_in, eye_pos, pitchyaw, length=40.0, thickness=2, color=(125, 125, 255)):
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)), tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color, thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out

if __name__ == '__main__':
    mtcnn_weights_dir = "./mtcnn_weights/"
    fd = MTCNNFaceDetector(sess=K.get_session(), model_path=mtcnn_weights_dir)

    model = KerasELG()
    model.net.load_weights("./elg_weights/elg_keras.h5")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024*3)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768*3)

    gaze_history = []
    while True:
        ret, input_img = cap.read()
        face, lms = fd.detect_face(input_img)

        if len(face) == 0 or len(face) > 1:
            continue
        else:
            print(">>> face detected.")

            '''draw the detected face.'''
            cv2.rectangle(input_img, (int(face[0][1]), int(face[0][0])), (int(face[0][3]), int(face[0][2])), (0, 255, 0), 2)

            '''detecting the left and right eye.'''
            left_eye_xy = np.array([lms[6], lms[1]])
            right_eye_xy = np.array([lms[5], lms[0]])

            dist_eyes = np.linalg.norm(left_eye_xy - right_eye_xy)
            eye_bbox_w = (dist_eyes/1.25)
            eye_bbox_h = (eye_bbox_w*0.6)

            left_eye_im = input_img[
                int(left_eye_xy[0]-eye_bbox_h//2):int(left_eye_xy[0]+eye_bbox_h//2),
                int(left_eye_xy[1]-eye_bbox_w//2):int(left_eye_xy[1]+eye_bbox_w//2), :]
            right_eye_im = input_img[
                int(right_eye_xy[0]-eye_bbox_h//2):int(right_eye_xy[0]+eye_bbox_h//2),
                int(right_eye_xy[1]-eye_bbox_w//2):int(right_eye_xy[1]+eye_bbox_w//2), :]

            # prepare a image.
            inp_left = cv2.cvtColor(left_eye_im, cv2.COLOR_RGB2GRAY)
            inp_left = cv2.equalizeHist(inp_left)
            inp_left = cv2.resize(inp_left, (180, 108))[np.newaxis, ..., np.newaxis]

            inp_right = cv2.cvtColor(right_eye_im, cv2.COLOR_RGB2GRAY)
            inp_right = cv2.equalizeHist(inp_right)
            inp_right = cv2.resize(inp_right, (180, 108))[np.newaxis, ..., np.newaxis]

            '''predict eye region landmarks.'''
            input_array = np.concatenate([inp_left, inp_right], axis=0)
            pred_left, pred_right = model.net.predict(input_array/255 * 2 - 1)

            '''right gaze estimation.'''
            lms_left = model._calculate_landmarks(pred_left)
            result_left, eyeball_landmark, _eyeball_centre, _iris_centre, eye_radius = draw_pupil(left_eye_im, inp_left, lms_left)

            eye_landmarks = np.concatenate([eyeball_landmark, [[eyeball_landmark[-1][0] + eye_radius/2, eyeball_landmark[-1][1]]]])
            eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)), 'constant', constant_values=1.0))
            eye_landmarks = np.asarray(eye_landmarks)
            eyelid_landmarks = eye_landmarks[0:8]
            iris_landmarks = eye_landmarks[8:16]
            iris_centre = eye_landmarks[16]
            eyeball_centre = eye_landmarks[17]
            eyeball_radius = np.linalg.norm(eye_landmarks[18] - eye_landmarks[17])
            print("eyeball_radius", eyeball_radius)
            i_x0, i_y0 = _iris_centre
            e_x0, e_y0 = _eyeball_centre

            theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
            phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)), -1.0, 1.0))
            current_gaze = np.array([theta, phi])
            gaze_history.append(current_gaze)

            max_len = 2
            if len(gaze_history) > max_len:
                gaze_history = gaze_history[-max_len:]
            draw_gaze(result_left, _iris_centre, np.mean(gaze_history, axis=0), length=250.0, thickness=2)

            # TODO: right gaze estimation.
            lms_right = model._calculate_landmarks(pred_right)
            result_right, _, _eyeball_centre, _iris_centre, _ = draw_pupil(right_eye_im, inp_right, lms_right)
            draw_gaze(result_right, _iris_centre, np.mean(gaze_history, axis=0), length=250.0, thickness=2)

            '''draw the eyes.'''
            slice_h = slice(int(left_eye_xy[0]-eye_bbox_h//2), int(left_eye_xy[0]+eye_bbox_h//2))
            slice_w = slice(int(left_eye_xy[1]-eye_bbox_w//2), int(left_eye_xy[1]+eye_bbox_w//2))
            im_shape = left_eye_im.shape[::-1]
            input_img[slice_h, slice_w, :] = cv2.resize(result_left, im_shape[1:])

            slice_h = slice(int(right_eye_xy[0]-eye_bbox_h//2), int(right_eye_xy[0]+eye_bbox_h//2))
            slice_w = slice(int(right_eye_xy[1]-eye_bbox_w//2), int(right_eye_xy[1]+eye_bbox_w//2))
            im_shape = right_eye_im.shape[::-1]
            input_img[slice_h, slice_w, :] = cv2.resize(result_right, im_shape[1:])
            cv2.imshow("show img", input_img)
            # cv2.waitKey(500)
            cv2.waitKey(1)