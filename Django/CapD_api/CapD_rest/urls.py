from django.urls import include, path
from rest_framework.routers import DefaultRouter
from . import views


router=DefaultRouter()
router.register('upload',views.UploadViewset,basename='upload')
router.register('detector',views.DetectorViewset,basename='detector')
router.register('non_idt',views.Non_idt_Viewset,basename='non_idt')
router.register('download',views.DownloadViewset,basename='download')
# router.register('mp4link',views.Download_linkViewset,basename='mp4lisk')

urlpatterns = [
    path('',include(router.urls)),
]