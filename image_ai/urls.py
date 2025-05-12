from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views
from .views import remove_watermark, background_removal

app_name = "image_ai"

urlpatterns = [
    path("", views.home, name="home"),
    path("crop/", views.crop, name="crop"),
    path("image-compression/", views.image_compression, name="image_compression"),
    path("about/", views.about, name="about"),
    path("passport-size-image/", views.passport_photo, name="passport_photo"),
    path("background_removal/", views.background_removal, name="background_removal"),
    path("download/<str:filename>/", views.download_image, name="download_image"),
    path('tools/passport_photo/', views.passport_photo, name='passport_photo'),
    path("enhancer/", views.enhancer, name="enhancer"),
    path("restore-photo/", views.restorePhoto, name="restorePhoto"),
    path("aiFilters/", views.aiFilters, name="aiFilters"),
    path('api/apply-filter/', views.apply_filter, name='apply-filter'),
    path('formate-convert/', views.formate_convert, name="Formate-Convert"),
    path('convert-file/', views.convert_file, name='convert_file'),
    path('watermark/', views.remove_watermark, name='watermark'),
    path('remove-watermark/', remove_watermark, name='remove_watermark'),
    path('background-removal/', background_removal, name='background_removal'),
    
] +  static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

