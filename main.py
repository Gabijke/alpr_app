import time
import cv2
import torch
import datetime
import threading
import argparse
import socket
from ocr_inference import predict
from multiprocessing import Queue
from db_conn import DataBase


def cam_read(url, y_start, y_end, x_start, x_end):

    model = torch.hub.load('./models/yolov5', 'custom', path='./models/yolov5m.pt', source='local')
    vcap = cv2.VideoCapture(url)
    width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vcap.get(cv2.CAP_PROP_FPS)
    resize_yA = int(y_start * height)
    resize_yB = int(y_end * height)
    resize_xA = int(x_start * width)
    resize_xB = int(x_end * width)

    key_frame = 1

    while vcap.isOpened():

        _ = vcap.grab()

        key_frame += 1

        if key_frame % fps == 0:

            _, frame = vcap.retrieve()

            try:
                resize_frame = frame[resize_yA: resize_yB, resize_xA: resize_xB]
                results = model(resize_frame)
            except TypeError:
                continue

            for box in results.xyxy[0]:

                if box[5] == 0:

                    xA = int(box[0])
                    yA = int(box[1])
                    xB = int(box[2])
                    yB = int(box[3])

                    box_img = resize_frame[yA:yB, xA:xB].copy()
                    img_with_box = cv2.rectangle(frame, (resize_xA + xA, resize_yA + yA),
                                                 (resize_xA + xB, resize_yA + yB), (0, 0, 255), thickness=2)
                    img_with_box = img_with_box[resize_yA: resize_yB, resize_xA: resize_xB]
                    q.put([img_with_box, box_img])


def predict_img():

    index = db.check_last_index()
    ip_localhost = socket.gethostbyname(socket.gethostname())

    while True:
        time.sleep(0.1)

        if not q.empty():

            full_img, box_img = q.get()

            label = predict(box_img)

            if label is not None:
                offset = datetime.timedelta(hours=5)
                time_event = datetime.datetime.now(datetime.timezone(offset)).strftime("%Y-%m-%d %H:%M:%S")

                filename = f"{label}.jpg"
                url_img = f"http://{ip_localhost}/images/{filename}"
                cv2.imwrite(f"final_segmentation/{filename}", full_img)
                item_tuple = (index, time_event, label, url_img)
                db.update_table(item_tuple)
                log_raw = str(index) + ". Распознан номер: " + label + ", время: " + time_event
                print(log_raw)
                index += 1


def argparse_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cam", type=str, help='rtsp stream', default='0')
    parser.add_argument("-y_start", type=float, help="perceptual field of the detector (start point on y-axis)", default=0.1)
    parser.add_argument("-y_end", type=float, help="perceptual field of the detector (end point on y-axis)", default=1.0)
    parser.add_argument("-x_start", type=float, help="perceptual field of the detector (start point on x-axis)", default=0.1)
    parser.add_argument("-x_end", type=float, help="perceptual field of the detector (end point on x-axis)", default=1.0)
    args = vars(parser.parse_args())
    return args


if __name__ == "__main__":
    db = DataBase(env_path=".env", table='cars_license_plate')
    args_dict = argparse_worker()
    cam = args_dict["cam"]
    y_start = args_dict["y_start"]
    y_end = args_dict["y_end"]
    x_start = args_dict["x_start"]
    x_end = args_dict["x_end"]
    q = Queue()
    p1 = threading.Thread(target=cam_read, args=(cam, x_start, x_end, y_start, y_end))
    p2 = threading.Thread(target=predict_img)
    p1.start()
    p2.start()
