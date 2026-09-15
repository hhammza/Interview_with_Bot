from django.urls import path

from . import views


urlpatterns = [
    path("health/", views.health),
    path("interviews/start/", views.start_interview),
    path("interviews/<uuid:session_id>/message/", views.send_message),
    path("interviews/<uuid:session_id>/face-frame/", views.analyse_face_frame),
    path("interviews/<uuid:session_id>/voice-sample/", views.analyse_voice_sample),
    path("interviews/<uuid:session_id>/recording/", views.upload_recording),
    path("interviews/<uuid:session_id>/finish/", views.finish_interview),
    path("interviews/<uuid:session_id>/transcript/", views.download_transcript),
    path("interviews/<uuid:session_id>/bundle/", views.download_bundle),
]
