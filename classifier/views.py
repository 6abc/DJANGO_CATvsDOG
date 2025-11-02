# classifier/views.py

import os
import numpy as np
import threading
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework import status
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from .models import TrainedModel, TrainingDataset, Prediction
from .forms import ModelCreateForm, DatasetUploadForm, PredictionForm, UserRegisterForm, UserLoginForm
import base64
from io import BytesIO

# ==================== Authentication Views ====================

def register_view(request):
    """User registration"""
    if request.user.is_authenticated:
        return redirect('index')
    
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            login(request, user)
            messages.success(request, f'Welcome {user.username}! Your account has been created.')
            return redirect('index')
    else:
        form = UserRegisterForm()
    
    return render(request, 'classifier/register.html', {'form': form})

def login_view(request):
    """User login"""
    if request.user.is_authenticated:
        return redirect('index')
    
    if request.method == 'POST':
        form = UserLoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(username=username, password=password)
            if user:
                login(request, user)
                next_url = request.GET.get('next', 'index')
                messages.success(request, f'Welcome back, {user.username}!')
                return redirect(next_url)
            else:
                messages.error(request, 'Invalid username or password')
    else:
        form = UserLoginForm()
    
    return render(request, 'classifier/login.html', {'form': form})

def logout_view(request):
    """User logout"""
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('login')

# ==================== Main Views ====================

def index(request):
    """Home page - list all models"""
    if request.user.is_authenticated:
        my_models = TrainedModel.objects.filter(user=request.user)
        shared_models = TrainedModel.objects.filter(is_public=True).exclude(user=request.user)
    else:
        my_models = []
        shared_models = TrainedModel.objects.filter(is_public=True)
    
    context = {
        'my_models': my_models,
        'shared_models': shared_models,
    }
    return render(request, 'classifier/index.html', context)

@login_required
def create_model(request):
    """Create a new model configuration"""
    if request.method == 'POST':
        form = ModelCreateForm(request.POST)
        if form.is_valid():
            model = form.save(commit=False)
            model.user = request.user
            model.save()
            messages.success(request, f'Model "{model.name}" created! Now upload training data.')
            return redirect('upload_data', model_id=model.id)
    else:
        form = ModelCreateForm()
    
    return render(request, 'classifier/create_model.html', {'form': form})

@login_required
def upload_data(request, model_id):
    """Upload training images for a model"""
    model = get_object_or_404(TrainedModel, id=model_id)
    
    # Check ownership
    if model.user != request.user:
        messages.error(request, 'You do not have permission to modify this model.')
        return redirect('index')
    
    if request.method == 'POST':
        category = request.POST.get('category')
        is_validation = request.POST.get('is_validation') == 'on'
        files = request.FILES.getlist('images')
        
        if not files:
            messages.error(request, 'Please select at least one image.')
        else:
            for file in files:
                TrainingDataset.objects.create(
                    trained_model=model,
                    category=category,
                    image=file,
                    is_validation=is_validation
                )
            messages.success(request, f'{len(files)} images uploaded successfully!')
            return redirect('upload_data', model_id=model.id)
    
    # Get statistics
    train_cats = model.datasets.filter(category='cats', is_validation=False).count()
    train_dogs = model.datasets.filter(category='dogs', is_validation=False).count()
    val_cats = model.datasets.filter(category='cats', is_validation=True).count()
    val_dogs = model.datasets.filter(category='dogs', is_validation=True).count()
    
    context = {
        'model': model,
        'train_cats': train_cats,
        'train_dogs': train_dogs,
        'val_cats': val_cats,
        'val_dogs': val_dogs,
        'total_train': train_cats + train_dogs,
        'total_val': val_cats + val_dogs,
    }
    
    return render(request, 'classifier/upload_data.html', context)

def train_model_in_background(model_id):
    """Background task to train the model"""
    try:
        model_obj = TrainedModel.objects.get(id=model_id)
        model_obj.status = 'training'
        model_obj.save()
        
        # Create temporary directories
        base_dir = os.path.join(settings.MEDIA_ROOT, 'temp_training', str(model_id))
        train_dir = os.path.join(base_dir, 'train')
        val_dir = os.path.join(base_dir, 'validation')
        
        os.makedirs(os.path.join(train_dir, 'cats'), exist_ok=True)
        os.makedirs(os.path.join(train_dir, 'dogs'), exist_ok=True)
        os.makedirs(os.path.join(val_dir, 'cats'), exist_ok=True)
        os.makedirs(os.path.join(val_dir, 'dogs'), exist_ok=True)
        
        # Copy images to temporary directories
        datasets = TrainingDataset.objects.filter(trained_model=model_obj)
        import shutil
        for dataset in datasets:
            if dataset.is_validation:
                dest_dir = os.path.join(val_dir, dataset.category)
            else:
                dest_dir = os.path.join(train_dir, dataset.category)
            
            src_path = dataset.image.path
            dest_path = os.path.join(dest_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dest_path)
        
        # Create data generators
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True
        )
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary'
        )
        
        validation_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary'
        )
        
        # Build model
        nn_model = Sequential([
            layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
            layers.MaxPooling2D(2,2),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D(2,2),
            layers.Conv2D(128, (3,3), activation='relu'),
            layers.MaxPooling2D(2,2),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        nn_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = nn_model.fit(
            train_generator,
            epochs=model_obj.epochs,
            validation_data=validation_generator,
            verbose=1
        )
        
        # Save model
        model_path = os.path.join(settings.MEDIA_ROOT, 'models', f'{model_id}.h5')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        nn_model.save(model_path)
        
        # Update model object
        model_obj.model_file = f'models/{model_id}.h5'
        model_obj.accuracy = float(history.history['accuracy'][-1]) * 100
        model_obj.val_accuracy = float(history.history['val_accuracy'][-1]) * 100
        model_obj.status = 'completed'
        model_obj.training_progress = 100
        model_obj.save()
        
        # Cleanup
        shutil.rmtree(base_dir)
        
    except Exception as e:
        model_obj.status = 'failed'
        model_obj.error_message = str(e)
        model_obj.save()

@login_required
def start_training(request, model_id):
    """Start training the model"""
    model = get_object_or_404(TrainedModel, id=model_id)
    
    # Check ownership
    if model.user != request.user:
        messages.error(request, 'You do not have permission to train this model.')
        return redirect('index')
    
    # Check if enough data
    train_count = model.datasets.filter(is_validation=False).count()
    val_count = model.datasets.filter(is_validation=True).count()
    
    if train_count < 20:
        messages.error(request, 'Please upload at least 10 images per category for training.')
        return redirect('upload_data', model_id=model.id)
    
    if val_count < 4:
        messages.error(request, 'Please upload at least 2 images per category for validation.')
        return redirect('upload_data', model_id=model.id)
    
    # Start training in background thread
    thread = threading.Thread(target=train_model_in_background, args=(model_id,))
    thread.daemon = True
    thread.start()
    
    messages.success(request, 'Training started! This may take several minutes.')
    return redirect('model_detail', model_id=model.id)

def model_detail(request, model_id):
    """View model details and training status"""
    model = get_object_or_404(TrainedModel, id=model_id)
    
    # Check access permissions
    if not model.is_public and model.user != request.user:
        messages.error(request, 'This model is private.')
        return redirect('index')
    
    predictions = model.predictions.all()[:10]
    
    context = {
        'model': model,
        'predictions': predictions,
        'is_owner': request.user == model.user if request.user.is_authenticated else False,
    }
    return render(request, 'classifier/model_detail.html', context)

@login_required
def toggle_model_visibility(request, model_id):
    """Toggle model public/private status"""
    model = get_object_or_404(TrainedModel, id=model_id)
    
    if model.user != request.user:
        messages.error(request, 'You do not have permission to modify this model.')
        return redirect('model_detail', model_id=model_id)
    
    model.is_public = not model.is_public
    model.save()
    
    status = "public" if model.is_public else "private"
    messages.success(request, f'Model is now {status}.')
    return redirect('model_detail', model_id=model_id)

def predict_image(request, model_id):
    """Make prediction using trained model"""
    model_obj = get_object_or_404(TrainedModel, id=model_id)
    
    # Check access permissions
    if not model_obj.is_public and (not request.user.is_authenticated or model_obj.user != request.user):
        messages.error(request, 'You do not have permission to use this model.')
        return redirect('index')
    
    if model_obj.status != 'completed':
        messages.error(request, 'This model is not ready for predictions yet.')
        return redirect('model_detail', model_id=model_id)
    
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
        
        try:
            # Save uploaded image
            prediction = Prediction.objects.create(
                trained_model=model_obj,
                image=uploaded_file,
                result='',
                confidence=0
            )
            
            # Load model and predict
            nn_model = load_model(model_obj.model_file.path)
            
            img = load_img(prediction.image.path, target_size=(150, 150))
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0
            
            pred = nn_model.predict(x, verbose=0)
            confidence = float(pred[0][0])
            
            if confidence > 0.5:
                result = "Dog"
                confidence_percent = confidence
            else:
                result = "Cat"
                confidence_percent = 1 - confidence
            
            # Update prediction
            prediction.result = result
            prediction.confidence = confidence_percent
            prediction.save()
            
            return render(request, 'classifier/prediction_result.html', {
                'model': model_obj,
                'prediction': prediction
            })
            
        except Exception as e:
            messages.error(request, f'Error making prediction: {str(e)}')
            return redirect('model_detail', model_id=model_id)
    
    return render(request, 'classifier/predict.html', {'model': model_obj})

def get_training_status(request, model_id):
    """API endpoint to check training status"""
    model = get_object_or_404(TrainedModel, id=model_id)
    return JsonResponse({
        'status': model.status,
        'progress': model.training_progress,
        'accuracy': model.accuracy,
        'val_accuracy': model.val_accuracy,
    })

@login_required
def my_models(request):
    """View user's own models"""
    models = TrainedModel.objects.filter(user=request.user)
    return render(request, 'classifier/my_models.html', {'models': models})

def public_models(request):
    """View all public models"""
    models = TrainedModel.objects.filter(is_public=True, status='completed')
    return render(request, 'classifier/public_models.html', {'models': models})

# ==================== REST API Views ====================

@api_view(['GET'])
@permission_classes([AllowAny])
def api_list_models(request):
    """API: List all public models"""
    models = TrainedModel.objects.filter(is_public=True, status='completed')
    data = [{
        'id': str(model.id),
        'name': model.name,
        'description': model.description,
        'accuracy': model.accuracy,
        'val_accuracy': model.val_accuracy,
        'created_at': model.created_at.isoformat(),
        'owner': model.user.username if model.user else 'Anonymous'
    } for model in models]
    
    return Response({'models': data, 'count': len(data)})

@api_view(['GET'])
@permission_classes([AllowAny])
def api_model_detail(request, model_id):
    """API: Get model details"""
    try:
        model = TrainedModel.objects.get(id=model_id)
        
        if not model.is_public and (not request.user.is_authenticated or model.user != request.user):
            return Response({'error': 'Model is private'}, status=status.HTTP_403_FORBIDDEN)
        
        data = {
            'id': str(model.id),
            'name': model.name,
            'description': model.description,
            'status': model.status,
            'accuracy': model.accuracy,
            'val_accuracy': model.val_accuracy,
            'epochs': model.epochs,
            'is_public': model.is_public,
            'created_at': model.created_at.isoformat(),
            'updated_at': model.updated_at.isoformat(),
            'owner': model.user.username if model.user else 'Anonymous',
            'predictions_count': model.predictions.count(),
        }
        
        return Response(data)
    except TrainedModel.DoesNotExist:
        return Response({'error': 'Model not found'}, status=status.HTTP_404_NOT_FOUND)

@api_view(['POST'])
@permission_classes([AllowAny])
def api_predict(request, model_id):
    """API: Make prediction with base64 image"""
    try:
        model_obj = TrainedModel.objects.get(id=model_id)
        
        if not model_obj.is_public and (not request.user.is_authenticated or model_obj.user != request.user):
            return Response({'error': 'Model is private'}, status=status.HTTP_403_FORBIDDEN)
        
        if model_obj.status != 'completed':
            return Response({'error': 'Model not ready'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Get base64 image from request
        image_data = request.data.get('image')
        if not image_data:
            return Response({'error': 'No image provided'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Decode base64 image
        try:
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = BytesIO(image_bytes)
        except Exception as e:
            return Response({'error': f'Invalid image data: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Load and preprocess image
        from PIL import Image
        img = Image.open(image)
        img = img.convert('RGB')
        img = img.resize((150, 150))
        
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0) / 255.0
        
        # Load model and predict
        nn_model = load_model(model_obj.model_file.path)
        pred = nn_model.predict(x, verbose=0)
        confidence = float(pred[0][0])
        
        if confidence > 0.5:
            result = "Dog"
            confidence_percent = confidence
        else:
            result = "Cat"
            confidence_percent = 1 - confidence
        
        return Response({
            'result': result,
            'confidence': confidence_percent * 100,
            'model_name': model_obj.name,
            'model_accuracy': model_obj.accuracy,
        })
        
    except TrainedModel.DoesNotExist:
        return Response({'error': 'Model not found'}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def api_my_models(request):
    """API: Get current user's models"""
    models = TrainedModel.objects.filter(user=request.user)
    data = [{
        'id': str(model.id),
        'name': model.name,
        'description': model.description,
        'status': model.status,
        'accuracy': model.accuracy,
        'val_accuracy': model.val_accuracy,
        'is_public': model.is_public,
        'created_at': model.created_at.isoformat(),
    } for model in models]
    
    return Response({'models': data, 'count': len(data)})