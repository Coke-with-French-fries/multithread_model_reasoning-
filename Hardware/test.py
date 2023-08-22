import cv2
import numpy as np
import onnxruntime as ort
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue

# 模型加载
model_pb_path = "v5lite-s.onnx"
so = ort.SessionOptions()
net = ort.InferenceSession(model_pb_path, so)

# 标签字典
dic_labels = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
              8: 'boat', 9: 'traffic light',
              10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
              16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
              20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
              27: 'tie', 28: 'suitcase', 29: 'frisbee',
              30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
              36: 'skateboard', 37: 'surfboard',
              38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
              45: 'bowl', 46: 'banana', 47: 'apple',
              48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
              55: 'cake', 56: 'chair', 57: 'couch',
              58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
              65: 'remote', 66: 'keyboard', 67: 'cell phone',
              68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock',
              75: 'vase', 76: 'scissors', 77: 'teddy bear',
              78: 'hair drier', 79: 'toothbrush'}

# 模型参数
model_h = 320
model_w = 320
nl = 3
na = 3
stride = [8., 16., 32.]
anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(nl, -1, 2)

flag_det = True

interval = 0

judge = 0

state = 0


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def _make_grid(nx, ny):
    xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
    return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)


def cal_outputs(outs, nl, na, model_w, model_h, anchor_grid, stride):
    row_ind = 0
    grid = [np.zeros(1)] * nl
    for i in range(nl):
        h, w = int(model_w / stride[i]), int(model_h / stride[i])
        length = int(na * h * w)
        if grid[i].shape[2:4] != (h, w):
            grid[i] = _make_grid(w, h)

        outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + np.tile(
            grid[i], (na, 1))) * int(stride[i])
        outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * np.repeat(
            anchor_grid[i], h * w, axis=0)
        row_ind += length
    return outs


def post_process_opencv(outputs, model_h, model_w, img_h, img_w, thred_nms, thred_cond):
    conf = outputs[:, 4].tolist()
    c_x = outputs[:, 0] / model_w * img_w
    c_y = outputs[:, 1] / model_h * img_h
    w = outputs[:, 2] / model_w * img_w
    h = outputs[:, 3] / model_h * img_h
    p_cls = outputs[:, 5:]
    if len(p_cls.shape) == 1:
        p_cls = np.expand_dims(p_cls, 1)
    cls_id = np.argmax(p_cls, axis=1)

    p_x1 = np.expand_dims(c_x - w / 2, -1)
    p_y1 = np.expand_dims(c_y - h / 2, -1)
    p_x2 = np.expand_dims(c_x + w / 2, -1)
    p_y2 = np.expand_dims(c_y + h / 2, -1)
    areas = np.concatenate((p_x1, p_y1, p_x2, p_y2), axis=-1)

    areas = areas.tolist()
    ids = cv2.dnn.NMSBoxes(areas, conf, thred_cond, thred_nms)
    if len(ids) > 0:
        return np.array(areas)[ids], np.array(conf)[ids], cls_id[ids]
    else:
        return [], [], []


def infer_img(img0, net, model_h, model_w, nl, na, stride, anchor_grid, thred_nms=0.4, thred_cond=0.5):
    # 图像预处理
    img = cv2.resize(img0, [model_w, model_h], interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    # 模型推理
    outs = net.run(None, {net.get_inputs()[0].name: blob})[0].squeeze(axis=0)

    # 输出坐标矫正
    outs = cal_outputs(outs, nl, na, model_w, model_h, anchor_grid, stride)

    # 检测框计算
    img_h, img_w, _ = np.shape(img0)
    boxes, confs, ids = post_process_opencv(outs, model_h, model_w, img_h, img_w, thred_nms, thred_cond)

    return boxes, confs, ids


# 进程1:处理当前帧
def process_current_frame(frame_queue, result_queue, net, model_h, model_w, nl, na, stride, anchor_grid):
    while True:
        frame = frame_queue.get()
        det_boxes, scores, ids = infer_img(frame, net, model_h, model_w, nl, na, stride, anchor_grid)

        result_queue.put((det_boxes, scores, ids))


# 进程2:处理下一帧
def process_next_frame(frame_queue, result_queue, net, model_h, model_w, nl, na, stride, anchor_grid):
    while True:
        frame = frame_queue.get()
        det_boxes, scores, ids = infer_img(frame, net, model_h, model_w, nl, na, stride, anchor_grid)

        result_queue.put((det_boxes, scores, ids))


def schedule(frame_queue, result_queue):
    cap = cv2.VideoCapture(0)
    while True:
        t1 = time.time()
        success, img0 = cap.read()
        if success:
            frame_queue.put(img0)

            success, img0 = cap.read()
            if success:
                frame_queue.put(img0)
                det_boxes, scores, ids = result_queue.get()
                for box, score, id in zip(det_boxes, scores, ids):
                    label = '%s:%.2f' % (dic_labels[id], score)
                    plot_one_box(box.astype(np.int16), img0, color=(255, 0, 0), label=label, line_thickness=None)
                cv2.imshow("video", img0)

                det_boxes, scores, ids = result_queue.get()
                for box, score, id in zip(det_boxes, scores, ids):
                    label = '%s:%.2f' % (dic_labels[id], score)
                    plot_one_box(box.astype(np.int16), img0, color=(255, 0, 0), label=label, line_thickness=None)
                cv2.imshow("video", img0)

        t2 = time.time()
        print((t2 - t1) / 2)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()


if __name__ == '__main__':
    pool = ThreadPoolExecutor(max_workers=4)

    frame_queue = Queue(maxsize=2)
    result_queue = Queue(maxsize=2)

    # 启动进程
    p1 = threading.Thread(target=process_current_frame,
                          args=(frame_queue, result_queue, net, model_h, model_w, nl, na, stride, anchor_grid))
    p2 = threading.Thread(target=process_next_frame,
                          args=(frame_queue, result_queue, net, model_h, model_w, nl, na, stride, anchor_grid))
    p3 = threading.Thread(target=schedule, args=(frame_queue, result_queue))

    p1.start()
    p2.start()
    p3.start()
