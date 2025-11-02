# classifier/models.py

from django.db import models
from django.contrib.auth.models import User
import uuid

class TrainedModel(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('training', 'Training'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    model_file = models.FileField(upload_to='models/', null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    accuracy = models.FloatField(null=True, blank=True)
    val_accuracy = models.FloatField(null=True, blank=True)
    epochs = models.IntegerField(default=10)
    training_progress = models.IntegerField(default=0)
    error_message = models.TextField(blank=True)
    is_public = models.BooleanField(default=False)  # New field for sharing
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} - {self.status}"

class TrainingDataset(models.Model):
    CATEGORY_CHOICES = [
        ('cats', 'Cats'),
        ('dogs', 'Dogs'),
    ]
    
    trained_model = models.ForeignKey(TrainedModel, on_delete=models.CASCADE, related_name='datasets')
    category = models.CharField(max_length=10, choices=CATEGORY_CHOICES)
    image = models.ImageField(upload_to='training_data/')
    is_validation = models.BooleanField(default=False)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['category', '-uploaded_at']
    
    def __str__(self):
        dataset_type = "Validation" if self.is_validation else "Training"
        return f"{self.trained_model.name} - {self.category} ({dataset_type})"

class Prediction(models.Model):
    trained_model = models.ForeignKey(TrainedModel, on_delete=models.CASCADE, related_name='predictions')
    image = models.ImageField(upload_to='predictions/')
    result = models.CharField(max_length=10)  # 'Cat' or 'Dog'
    confidence = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.result} - {self.confidence:.2%}"