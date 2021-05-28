from django.db.models.query import QuerySet
from rest_framework import mixins
from rest_framework.viewsets import ModelViewSet
from rest_framework.response import Response
from .serializers import UploadSerializer,PersonSerializer,Selected_PersonSerializer
from .models import Video,Person,Selected_Person


from glob import glob
from .app.detector import Detector
from .app.non_identification import Non_idt
from .app.save_output import Saving
# from .app.main import *
from django.contrib.staticfiles.storage import staticfiles_storage



class UploadViewset(ModelViewSet):
    queryset=Video.objects.all()
    serializer_class = UploadSerializer
    #post
    def create(self, request):
            data=request.data
            model_video=UploadSerializer(data=data)
            if model_video.is_valid():
                model_video.save()
                ret=UploadSerializer(Video.objects.all(),many=True)
                return Response(ret.data)
            else:
                return Response(status=404)

    #get
    def list(self,request):
        serializer = UploadSerializer(Video.objects.all(), many=True)
        return Response(serializer.data)

ret_bboxss=None
ret_identitiess=None
ret_img=None
cap=None

class DetectorViewset(ModelViewSet):
    queryset=Person.objects.all()
    serializer_class=PersonSerializer

    def _detecting(self):
        global ret_bboxss
        global ret_identitiess
        global ret_img
        global cap
        t1=Detector('CapD_rest/app/source/test_short.mp4')
        ret_bboxss,ret_identitiess,ret_img,cap=t1.yolo_deep_det()


    def list(self,request):

        self._detecting()

        one_cap_path='CapD_rest/app/one_cap'
        condition='/*.jpg'
        Qset=list(Person.objects.all().values('det_id'))
        db_id_list=[]
        file_id_list=[]
        for dic in Qset:
            db_id_list.append(int(dic['det_id']))
    
        for img_path in glob(one_cap_path+condition):
            print(img_path)
            file_id=img_path.split('/')[-1]
            file_name=file_id
            file_id=int(file_id.split('.')[0])
            file_id_list.append(file_id)
            
            url=request.build_absolute_uri(staticfiles_storage.url(file_name))
            # print(url)
    
            if file_id not in db_id_list:
                Person.objects.create(det_id=file_id,img_url=url)

        for db_id in db_id_list:
            if db_id not in file_id_list:
                Person.objects.filter(det_id=db_id).delete()  
 
        img_db=PersonSerializer(Person.objects.all(),many=True)
        return Response(img_db.data)



class Non_idt_Viewset(ModelViewSet):
    queryset=Selected_Person.objects.all()
    serializer_class=Selected_PersonSerializer
        #select ids
    def create(self, request):
        #[1,2,3,4,5] : input form
        data=request.data
        Selected_Person_model=Selected_PersonSerializer(data=data)

        # store only one value on models
        if Selected_Person_model.is_valid():
            ret=Selected_PersonSerializer(Selected_Person.objects.all(),many=True)
            if Selected_Person.objects.exists():
                Selected_Person.objects.all().delete()
                Selected_Person_model.save()
            else:
                Selected_Person_model.save()
            return Response(ret.data)
        else:
            return Response(status=404)

    

        
   
