import shutil
import os

from rest_framework.viewsets import ModelViewSet
from rest_framework.response import Response
from django.http import FileResponse
from .serializers import UploadSerializer,PersonSerializer,Selected_PersonSerializer,DownloadSerializer
from .models import Video,Person,Selected_Person,Download
from rest_framework import status

from glob import glob
from .app.detector import Detector
from .app.non_identification import Non_idt
from .app.save_output import Saving

from django.contrib.staticfiles.storage import staticfiles_storage

ret_bboxss=None
ret_identitiess=None
ret_img=None
cap=None
processing_imgs=None
file_name=None

class UploadViewset(ModelViewSet):
    queryset=Video.objects.all()
    serializer_class = UploadSerializer
    #post
    def create(self, request):
            global file_name

            serializer = self.get_serializer(data=request.data)
            vid_name=dict(request.data)
            if vid_name:
                vid_name=vid_name['video_file']
                vid_name=str(vid_name).split(' ')[1]
                file_name=vid_name

            if os.path.exists('CapD_rest/app/source/'):
                shutil.rmtree('CapD_rest/app/source/')
                os.mkdir('CapD_rest/app/source/')
            if serializer.is_valid(raise_exception=True):
                self.perform_create(serializer)
                headers = self.get_success_headers(serializer.data)
                return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
            else:
                headers = self.get_success_headers(serializer.data)
                return Response(status=status.HTTP_400_BAD_REQUEST,headers=headers)

    #get
    def list(self,request):
        queryset = self.filter_queryset(self.get_queryset())
        
        if queryset.count()>10:
            queryset.delete()

        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

class DetectorViewset(ModelViewSet):
    queryset=Person.objects.all()
    serializer_class=PersonSerializer

    def _detecting(self):
        global ret_bboxss
        global ret_identitiess
        global ret_img
        global cap
        global file_name
        
        t1=Detector('CapD_rest/app/source/'+file_name)
        file_name=t1.get_name()
        ret_bboxss,ret_identitiess,ret_img,cap=t1.yolo_deep_det()

    def list(self,request):
        self._detecting()

        one_cap_path='CapD_rest/app/one_cap'
        condition='/*.jpg'
        Qset=list(Person.objects.all().values('det_id'))
        db_id_list=[]
        img_id_list=[]
        for dic in Qset:
            db_id_list.append(int(dic['det_id']))
    
        for img_path in glob(one_cap_path+condition):
            print(img_path)
            img_id=img_path.split('/')[-1]
            img_name=img_id
            img_id=int(img_id.split('.')[0])
            img_id_list.append(img_id)
            
            url=request.build_absolute_uri(staticfiles_storage.url(img_name))
            # print(url)
    
            if img_id not in db_id_list:
                Person.objects.create(det_id=img_id,img_url=url)

        for db_id in db_id_list:
            if db_id not in img_id_list:
                Person.objects.filter(det_id=db_id).delete()  

        queryset = self.filter_queryset(self.get_queryset())
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

class Non_idt_Viewset(ModelViewSet):
    queryset=Selected_Person.objects.all()
    serializer_class=Selected_PersonSerializer
        #select ids
    def create(self, request):
        #1,2,3,4,5 : input form
        # data=request.data
        # Selected_Person_model=Selected_PersonSerializer(data=data)
        serializer = self.get_serializer(data=request.data)
        # serializer.is_valid(raise_exception=True)
        # store only one value on models
        # if Selected_Person_model.is_valid():
        #     # ret=Selected_PersonSerializer(Selected_Person.objects.all(),many=True)
        if serializer.is_valid(raise_exception=True):
            if Selected_Person.objects.exists():
                # same as queryset.delete()
                Selected_Person.objects.all().delete()
                
                self.perform_create(serializer)
            else:
                self.perform_create(serializer)

            headers = self.get_success_headers(serializer.data)
            return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)
        else:
            headers = self.get_success_headers(serializer.data)
            return Response(status=status.HTTP_400_BAD_REQUEST,headers=headers)

    
class DownloadViewset(ModelViewSet):
    queryset=Download.objects.all()
    serializer_class=DownloadSerializer

    def _downloading(self,selected_id):
        global processing_imgs
        # print(processing_imgs)
        t2=Non_idt(ret_bboxss,ret_identitiess,ret_img)
        processing_imgs=t2.non_idt_func(selected_id)
        
        t3=Saving(processing_imgs)
        t3.res_save(cap,file_name)

    def list(self,request):
        global file_name
        selected_person_list=list(Selected_Person.objects.all().values('selected_list'))
        selected_person_list=selected_person_list[0]['selected_list']
        selected_person_list=list(map(int,selected_person_list.replace('_',',').split(',')))
        Download.objects.all().delete()

        self._downloading(selected_person_list)

        file_name='output_'+file_name
        if os.path.exists('CapD_rest/app/output_vid/'+file_name):
            download_url=request.build_absolute_uri(staticfiles_storage.url(file_name))
            print(download_url)
            Download.objects.create(url=download_url)

            #---------test file download
            file_path='CapD_rest/app/output_vid/'+file_name
            response = FileResponse(open(file_path, 'rb'), content_type="mp4")
            #Content-Disposition 에서 file name field check
            response['Content-Disposition'] = 'attachment; filename='+file_name
            
            return response

        # queryset = self.filter_queryset(self.get_queryset())
        # page = self.paginate_queryset(queryset)
        # if page is not None:
        #     serializer = self.get_serializer(page, many=True)
        #     return self.get_paginated_response(serializer.data)
            
        # serializer = self.get_serializer(queryset, many=True)
        # return Response(serializer.data)

   
