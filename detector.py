import sys
import cv2
import torch

sys.path.insert(0, './yolov5')
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
from pathlib import Path



#color boundery
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def compute_color_for_labels(label):
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def mosaic(img,ratio=0.1):
    small=cv2.resize(img,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small,img.shape[:2][::-1],interpolation=cv2.INTER_NEAREST)

def area_mosaic(img,x1,y1,x2,y2,ratio=0.1):
    left=min(x1,x2)
    top=min(y1,y2)
    width=abs(x1-x2)
    height=abs(y1-y2)
    img[top:top + height, left:left + width] = mosaic(img[top:top + height, left:left + width], ratio)
    return img
    

def draw_boxes(img, bbox,confs, identities=None, offset=(0, 0)):
    
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        try:
          conf='%.2f'%confs[i][0]
        except IndexError:
          conf=-1
        
        color = compute_color_for_labels(id)
        label = '{}id:{} :{}'.format("", id,conf)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 1)[0]
        area_mosaic(img,x1,y1,x2,y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def yolo_2_deep(*xyxy):
    
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h

    return x_c, y_c, w, h

def yolo_deep_det(opt):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    #init
    device = select_device(opt.device)
    half = device.type != 'cpu'

    #init deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    #load model
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    view_img = True
    save_img = True
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    #output path
    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'

    for frame_idx,(path,img,im0s,vid_cap)in enumerate(dataset):
        img=torch.from_numpy(img).to(device)
        img=img.half() if half else img.float()
        img/=255.0
        img=img.unsqueeze(0) if img.ndimension()==3 else img
        ret_txt=''

        #inference
        t1=time_synchronized()
        pred=model(img,augment=opt.augment)[0]

        #Apply NMS
        pred=non_max_suppression(pred,conf_thres=0.4,iou_thres=0.5,classes=0)
        t2=time_synchronized()

        #process detection
        for i,det in enumerate(pred):
            ret_str=''
            save_path = str(Path(out) / Path(path).name)
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    ret_str += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []

                #Adapt deepsort format
                for *xyxy,conf,cls in det:
                    x_c,y_c,bbox_w,bbox_h=yolo_2_deep(*xyxy)
                    obj=[x_c,y_c,bbox_w,bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs=torch.tensor(bbox_xywh)
                confss=torch.tensor(confs)
                
                #deepsorting
                outputs=deepsort.update(xywhs,confss,im0s)

                #draw bbox
                if len(outputs)>0:
                    bbox_xyxy=outputs[:,:4]
                    identities=outputs[:,-1]
                    draw_boxes(im0s,bbox_xyxy,confs,identities)

                #save_txt
                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_right = output[2]
                        bbox_bottom = output[3]
                        identity = output[-1]
                        try:
                            conf='%.2f'%confs[identity][0]
                        except IndexError:
                            conf=0

                        # with open(txt_path, 'w') as f:
                        #     f.write(('%g ' * 6 + '\n') % (frame_idx, identity, bbox_left, bbox_top, bbox_right, bbox_bottom))  # label format
                        ret_txt+='{} {} {} {} {} {}\n'.format(identity, conf, bbox_left, bbox_top, bbox_right, bbox_bottom)
                        
            else:
                deepsort.increment_ages()

            #make map form txt
            with open(str(Path(out))+'\\'+'%07d'%frame_idx+'.txt', 'w') as f:
                f.write(ret_txt)
    
            print('%sDone. (%.3fs)' % (ret_str, t2 - t1))

            # Stream results---------------------------------------------------------------------
            if view_img:
                cv2.imshow(path, im0s)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            if dataset.mode == 'image':
                print('saving img!')
            else:
                # Save results (image with detections)
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(
                        save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                vid_writer.write(im0s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/yolov5x.pt', help='model.pt path')
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', default=False,
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')    
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        yolo_deep_det(args)