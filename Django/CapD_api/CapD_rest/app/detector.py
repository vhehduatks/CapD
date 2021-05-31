import sys
import os
import torch
import cv2
import shutil

# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.insert(0, 'CapD_rest/app/yolov5')
from .yolov5.utils.datasets import LoadImages
from .yolov5.utils.general import  non_max_suppression, scale_coords
from .yolov5.utils.torch_utils import select_device, time_synchronized
from .deep_sort_pytorch.utils.parser import get_config
from .deep_sort_pytorch.deep_sort import DeepSort
from pathlib import Path


class Detector:
    def __init__(self,source,device='cpu',weights='CapD_rest/app/yolov5/weights/yolov5s.pt'):
        self.out='CapD_rest/app/output'
        self.source=source
        self.save_txt=True
        self.imgsz=640
        self.augment=True
        self.device=device
        self.weights=weights

        #init deepsort
        self.cfg = get_config()
        print(os.path.relpath( os.path.dirname(os.path.abspath(os.path.dirname(__file__))) ))
        self.cfg.merge_from_file('CapD_rest/app/deep_sort_pytorch/configs/deep_sort.yaml')
        self.deepsort = DeepSort(self.cfg.DEEPSORT.REID_CKPT,
                            max_dist=self.cfg.DEEPSORT.MAX_DIST, 
                            min_confidence=self.cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=self.cfg.DEEPSORT.NMS_MAX_OVERLAP, 
                            max_iou_distance=self.cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=self.cfg.DEEPSORT.MAX_AGE, 
                            n_init=self.cfg.DEEPSORT.N_INIT, 
                            nn_budget=self.cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        
        #set device,dataset
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'
        self.vid_path = None
        self.vid_writer = None
        self.dataset = LoadImages(self.source, img_size=self.imgsz)
        
        #init yolo

        self.model=torch.load(self.weights,map_location=self.device)['model'].float()
        self.model.to(self.device).eval()
        if self.half:
            self.model.half()
        
        #get name
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        
        #else 
        self.xywhs=None
        self.confss=None
        self.im0s=None
        self.one_cap_path='CapD_rest/app/one_cap/'
        self.output_vid_path='CapD_rest/app/output_vid'
        self._clear_dir()
        
    def yolo_deep_det(self):
        ret_bbox=list()
        ret_img=list()
        ret_identities=list()
        have_pic=list()
        cap=None
      
        for frame_idx,(path,img,im0s,vid_cap)in enumerate(self.dataset):
         
            img=torch.from_numpy(img).to(self.device)
            img=img.half() if self.half else img.float()
            img/=255.0
            img=img.unsqueeze(0) if img.ndimension()==3 else img
            self.im0s=im0s
            
            #inference
            t1=time_synchronized()
            pred=self.model(img,augment=self.augment)[0]

            #Apply NMS
            pred=non_max_suppression(pred,conf_thres=0.4,iou_thres=0.5,classes=0)
            t2=time_synchronized()

            #process detection
            for i,det in enumerate(pred):
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], self.im0s.shape).round()
                    bbox_xywh = []
                    confs = []
                    det_info=''

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        det_info += '{} {}s, '.format(n, self.names[int(c)])  # add to string

                    #Adapt deepsort format
                    for *xyxy,conf,cls in det:
                        x_c,y_c,bbox_w,bbox_h=self._yolo_2_deep(*xyxy)
                        obj=[x_c,y_c,bbox_w,bbox_h]
                        bbox_xywh.append(obj)
                        confs.append([conf.item()])

                    self.xywhs=torch.tensor(bbox_xywh)
                    self.confss=torch.tensor(confs)

                    #deepsorting
                    outputs=self.deepsort.update(self.xywhs,self.confss,self.im0s)
                    
                    for output in outputs:
                        self._one_capture(output,im0s,have_pic)    
                    ret_img.append(im0s)
                    # print(frame_idx,'out:',outputs,im0s)
                    
                    if len(outputs)>0:
                        ret_bbox.append(outputs[:,:4])
                        ret_identities.append(outputs[:,-1])
                    self._save_txt(self.save_txt,outputs,confs,frame_idx)
                else:
                    self.deepsort.increment_ages()

                print('Find Objects :{} ({}s)'.format(det_info,'%.3f'%(t2-t1)))

            if not cap:
                cap=self._get_fps_w_h(vid_cap)

        return ret_bbox,ret_identities,ret_img,cap

    def get_name(self):
        return Path(self.source).name

    def _save_txt(self,txt_bool,outputs,confs,frame_idx):
        ret_txt=''
        if txt_bool and len(outputs) != 0:
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
                ret_txt+='{} {} {} {} {} {}\n'.format(identity, conf, bbox_left, bbox_top, bbox_right, bbox_bottom)

        if txt_bool:
            with open(str(Path(self.out))+'//'+'%07d'%frame_idx+'.txt', 'w') as f:
                f.write(ret_txt)

    def _clear_dir(self):
        if os.path.exists(self.out):
            shutil.rmtree(self.out)  # delete output folder
        if os.path.exists(self.one_cap_path):
            shutil.rmtree(self.one_cap_path)  # delete one_cap_path folder
        if os.path.exists(self.output_vid_path):
            shutil.rmtree(self.output_vid_path)  # delete one_cap_path folder
        os.makedirs(self.output_vid_path)
        os.makedirs(self.out)
        os.makedirs(self.one_cap_path)

    def _get_fps_w_h(self,vid_cap):
        return vid_cap.get(cv2.CAP_PROP_FPS),int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _one_capture(self,deep_res,img,have_pic,path=None):
        # img[top:top + height, left:left + width]
        path=self.one_cap_path
        bbox_left = deep_res[0]
        bbox_top = deep_res[1]
        bbox_right = deep_res[2]
        bbox_bottom = deep_res[3]
        identity = deep_res[-1]
        if identity in have_pic:
            pass
        else:
            have_pic.append(identity)
            img=img[bbox_top:bbox_bottom, bbox_left:bbox_right]
            cv2.imwrite(path+'%04d'%identity+'.jpg',img)
 
    def _yolo_2_deep(self,*xyxy):
            
        bbox_left = min([xyxy[0].item(), xyxy[2].item()])
        bbox_top = min([xyxy[1].item(), xyxy[3].item()])
        bbox_w = abs(xyxy[0].item() - xyxy[2].item())
        bbox_h = abs(xyxy[1].item() - xyxy[3].item())
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h

        return x_c, y_c, w, h

        

# output : ret_bbox,ret_identities,ret_img,cap,bbox_txt,bbox_img
                              

