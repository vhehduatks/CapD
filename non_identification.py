import cv2

#color boundery
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

class Non_idt:
    def __init__(self, ret_bboxss,ret_identitiess,ret_imgs):
        self.bboxss = ret_bboxss
        self.idss=ret_identitiess
        self.imgs = ret_imgs

    def non_idt(self,selects):
        ret=list()
        for bboxs,ids,img in zip(self.bboxss,self.idss,self.imgs):
            ret_img=self._processing(selects,img,bboxs,ids)
            ret.append(ret_img)

        return ret
        
    def _processing(self,selects,img,bbox, identities=None, offset=(0, 0)):
        
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0

            if self._select(id,selects):
                color = self._compute_color_for_labels(id)
                label = '{}id:{}'.format("", id)
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 1)[0]
                self._area_mosaic(img,x1,y1,x2,y2)
                #------remove----
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
                cv2.putText(img, label, (x1, y1 +
                                        t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
                #----------------
        return img
            
    def _select(self,id,selected_id):
        return False if id in selected_id else True

    def _compute_color_for_labels(self,label):
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return tuple(color)

    def _mosaic(self,img,ratio=0.1):
        small=cv2.resize(img,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_NEAREST)
        return cv2.resize(small,img.shape[:2][::-1],interpolation=cv2.INTER_NEAREST)

    def _area_mosaic(self,img,x1,y1,x2,y2,ratio=0.1):
        left=min(x1,x2)
        top=min(y1,y2)
        width=abs(x1-x2)
        height=abs(y1-y2)
        img[top:top + height, left:left + width] = self._mosaic(img[top:top + height, left:left + width], ratio)
        return img
        
        

# output : non identified images
                                
    