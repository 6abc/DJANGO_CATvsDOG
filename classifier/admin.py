from django.contrib import admin
from .models import TrainedModel, TrainingDataset, Prediction
# Register your models here.
admin.site.register(TrainedModel)
admin.site.register(TrainingDataset)
admin.site.register(Prediction)