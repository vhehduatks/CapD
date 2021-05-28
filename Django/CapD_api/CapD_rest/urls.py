from django.urls import include, path
from rest_framework.routers import DefaultRouter
from . import views


router=DefaultRouter()
router.register('upload',views.UploadViewset,basename='upload')
router.register('detector',views.DetectorViewset,basename='detector')

urlpatterns = [
    path('',include(router.urls)),
]