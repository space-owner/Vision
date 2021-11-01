import os
import time
import cv2

input_path = os.getcwd().replace("\\", "/") + "/data/input"
print(">>> data directory =", input_path)

files = os.listdir(input_path)
print(">>> total file list =", files, end="\n"*2)

for f in files:
    print(">>> reading the file =", input_path + "/" + f)
    cap = cv2.VideoCapture(input_path + "/" + f)
    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                cap.release()
                cv2.destroyAllWindows()
            cv2.imshow("video", frame)

            key = cv2.waitKey(10)

            if key==49:
                cap.release()
                cv2.destroyAllWindows()
                break
    except:
        print(">>> loading the next ...")
        time.sleep(3)
        continue