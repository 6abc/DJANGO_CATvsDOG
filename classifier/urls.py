# classifier/urls.py

from django.urls import path
from . import views

urlpatterns = [
    # Authentication
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    
    # Main pages
    path('', views.index, name='index'),
    path('my-models/', views.my_models, name='my_models'),
    path('public-models/', views.public_models, name='public_models'),
    
    # Model management
    path('create/', views.create_model, name='create_model'),
    path('model/<uuid:model_id>/', views.model_detail, name='model_detail'),
    path('model/<uuid:model_id>/upload/', views.upload_data, name='upload_data'),
    path('model/<uuid:model_id>/train/', views.start_training, name='start_training'),
    path('model/<uuid:model_id>/predict/', views.predict_image, name='predict_image'),
    path('model/<uuid:model_id>/toggle-visibility/', views.toggle_model_visibility, name='toggle_visibility'),
    
    # AJAX endpoints
    path('api/model/<uuid:model_id>/status/', views.get_training_status, name='training_status'),
    
    # REST API endpoints
    path('api/models/', views.api_list_models, name='api_list_models'),
    path('api/models/my/', views.api_my_models, name='api_my_models'),
    path('api/model/<uuid:model_id>/', views.api_model_detail, name='api_model_detail'),
    path('api/model/<uuid:model_id>/predict/', views.api_predict, name='api_predict'),
]