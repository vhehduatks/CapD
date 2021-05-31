import argparse

from .non_identification import Non_idt
from .detector import Detector
from .save_output import Saving

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str,
                    default='yolov5/weights/yolov5s.pt', help='model.pt path')
parser.add_argument('--source', type=str,
                    default='test_short.mp4', help='source')
parser.add_argument('--output', type=str, default='output',
                    help='output folder')  # output folder
parser.add_argument('--img-size', type=int, default=640,
                    help='inference size (pixels)')
parser.add_argument('--fourcc', type=str, default='mp4v',
                    help='output video codec (verify ffmpeg support)')
parser.add_argument('--device', default='',
                    help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--save_txt', action='store_true',
                    help='save results to *.txt')
parser.add_argument('--augment', action='store_true',
                    help='augmented inference')    
parser.add_argument("--config_deepsort", type=str,
                    default="deep_sort_pytorch/configs/deep_sort.yaml")
parser.add_argument("--file_name", type=str,
                    default="None")
args = parser.parse_args()

# class Main:

#     def Detector_class(self):
#         detector=Detector(args)    
#         ret_bboxss,ret_identitiess,ret_img,cap=detector.yolo_deep_det()
#         args.file_name=detector.get_name()

#         return ret_bboxss,ret_identitiess,ret_img,cap

#     def Non_idt_class(self,ret_bboxss,ret_identitiess,ret_img,selected_ids):
#         non_idt=Non_idt(ret_bboxss,ret_identitiess,ret_img)
#         processing_imgs=non_idt.non_idt_func(selected_ids)
#         return processing_imgs

#     def Saving(self,processing_imgs,cap):
#         saving=Saving(processing_imgs,args.output,args.fourcc)
#         saving.res_save(cap,args.file_name)


# t1=Detector(args)    
# ret_bboxss,ret_identitiess,ret_img,cap=t1.yolo_deep_det()
# file_name=t1.get_name()

# t2=Non_idt(ret_bboxss,ret_identitiess,ret_img)
# selected_id=[]
# processing_imgs=t2.non_idt_func(selected_id)

# t3=Saving(processing_imgs,args.output,args.fourcc)
# t3.res_save(cap,file_name)





