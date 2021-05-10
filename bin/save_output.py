import cv2


class Saving:

    def __init__(self, imgs, path, vid_codec):
        self.imgs = imgs
        self.path = path
        self.fourcc=vid_codec
        self.vid_path=None
        self.vid_writer=None

    def res_save(self,vid_cap,file_name):
        fps,w,h=vid_cap
        for i,im0s in enumerate(self.imgs):

            if self.vid_path != self.path:  # new video
                self.vid_path = self.path
                if isinstance(self.vid_writer, cv2.VideoWriter):
                    self.vid_writer.release()  # release previous video writer
                self.vid_writer = cv2.VideoWriter(self.path+'/output_'+file_name, cv2.VideoWriter_fourcc(*self.fourcc), fps, (w, h))
            self.vid_writer.write(im0s)
            print('save vid frame({}/{})'.format(i+1,len(self.imgs)))