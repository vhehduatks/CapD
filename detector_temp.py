
import sys
import torch

sys.path.insert(0, './yolov5')
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import  non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

"""
detector 클래스는 탐색+ 추적만
def yolo_deep_det(opt):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
"""

class Detector:
    def __init__(self,opt):
        self.out=opt.output
        self.source=opt.source
        self.save_txt=opt.save_txt
        self.imgsz=opt.img_size
        self.augment=opt.augment

        #init deepsort
        self.cfg = get_config()
        self.cfg.merge_from_file(opt.config_deepsort)
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
        self.device = select_device(opt.device)
        self.half = self.device.type != 'cpu'
        self.vid_path = None
        self.vid_writer = None
        self.view_img = opt.view_img
        print(self.source)
        self.dataset = LoadImages(self.source, img_size=self.imgsz)
        
        #init yolo
        print(opt.weights)
        self.model=torch.load(opt.weights,map_location=self.device)['model'].float()
        self.model.to(self.device).eval()
        if self.half:
            self.model.half()
        
        #get name
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        
        #else 
        self.xywhs=None
        self.confss=None
        self.im0s=None
        print("init ok")
        
    def yolo_deep_det(self):
        print('in det')
        ret_bbox=list()
        ret_img=list()
        ret_identities=list()
        for frame_idx,(path,img,im0s,vid_cap)in enumerate(self.dataset):
            img=torch.from_numpy(img).to(self.device)
            img=img.half() if self.half else img.float()
            img/=255.0
            img=img.unsqueeze(0) if img.ndimension()==3 else img
            self.im0s=im0s
            ret_txt=''

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
                    
                        
                    ret_img.append(im0s)
                    # print(frame_idx,'out:',outputs,im0s)
                    
                    if len(outputs)>0:
                        ret_bbox.append(outputs[:,:4])
                        ret_identities.append(outputs[:,-1])
                else:
                    self.deepsort.increment_ages()

        return ret_bbox,ret_identities,ret_img        
        
    
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

        
"""
 outputs=[[left,top,right,bottom,identities],
            [left,top,right,bottom,identities],
            ,,,
            [left,top,right,bottom,identities]]
"""                                 

