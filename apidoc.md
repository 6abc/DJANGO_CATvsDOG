# REST API Documentation

Complete API reference for the Cat vs Dog Classifier platform.

## Base URL
```
http://127.0.0.1:8000/api/
```

## Authentication

Most endpoints support both authenticated and unauthenticated access:
- **Session Authentication**: Use Django's built-in session authentication (login via web interface)
- **Basic Authentication**: Send username/password with each request

### Example with Basic Auth:
```bash
curl -u username:password http://127.0.0.1:8000/api/models/
```

---

## Endpoints

### 1. List All Public Models

Get a list of all publicly available trained models.

**Endpoint:** `GET /api/models/`  
**Authentication:** Not required  
**Permissions:** Public access

#### Request Example:
```bash
curl http://127.0.0.1:8000/api/models/
```

#### Response Example:
```json
{
  "models": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "High Accuracy Cat-Dog Classifier",
      "description": "Trained with 1000+ images, 95% accuracy",
      "accuracy": 95.5,
      "val_accuracy": 93.2,
      "created_at": "2025-11-03T10:30:00Z",
      "owner": "john_doe"
    },
    {
      "id": "660e8400-e29b-41d4-a716-446655440001",
      "name": "Quick Classifier",
      "description": "Fast training model for testing",
      "accuracy": 88.0,
      "val_accuracy": 85.5,
      "created_at": "2025-11-02T15:20:00Z",
      "owner": "jane_smith"
    }
  ],
  "count": 2
}
```

---

### 2. Get Model Details

Get detailed information about a specific model.

**Endpoint:** `GET /api/model/{model_id}/`  
**Authentication:** Required for private models  
**Permissions:** Owner or public access

#### Request Example:
```bash
curl http://127.0.0.1:8000/api/model/550e8400-e29b-41d4-a716-446655440000/
```

#### Response Example:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "High Accuracy Cat-Dog Classifier",
  "description": "Trained with 1000+ images, 95% accuracy",
  "status": "completed",
  "accuracy": 95.5,
  "val_accuracy": 93.2,
  "epochs": 15,
  "is_public": true,
  "created_at": "2025-11-03T10:30:00Z",
  "updated_at": "2025-11-03T11:45:00Z",
  "owner": "john_doe",
  "predictions_count": 127
}
```

---

### 3. Make Prediction (Base64 Image)

Use a trained model to classify an image.

**Endpoint:** `POST /api/model/{model_id}/predict/`  
**Authentication:** Not required for public models  
**Permissions:** Owner or public access  
**Content-Type:** `application/json`

#### Request Body:
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```

Or simply:
```json
{
  "image": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```

#### Request Example (Python):
```python
import requests
import base64

# Read and encode image
with open('cat.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# Make prediction
response = requests.post(
    'http://127.0.0.1:8000/api/model/550e8400-e29b-41d4-a716-446655440000/predict/',
    json={'image': image_data}
)

print(response.json())
```

#### Request Example (JavaScript):
```javascript
// Convert file to base64
const fileToBase64 = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.onerror = error => reject(error);
  });
};

// Make prediction
const file = document.querySelector('input[type="file"]').files[0];
const base64Image = await fileToBase64(file);

fetch('http://127.0.0.1:8000/api/model/550e8400-e29b-41d4-a716-446655440000/predict/', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ image: base64Image })
})
.then(response => response.json())
.then(data => console.log(data));
```

#### Request Example (cURL):
```bash
# Encode image to base64
IMAGE_BASE64=$(base64 -w 0 cat.jpg)

# Make prediction
curl -X POST http://127.0.0.1:8000/api/model/550e8400-e29b-41d4-a716-446655440000/predict/ \
  -H "Content-Type: application/json" \
  -d "{\"image\":\"$IMAGE_BASE64\"}"
```

#### Response Example:
```json
{
  "result": "Cat",
  "confidence": 94.5,
  "model_name": "High Accuracy Cat-Dog Classifier",
  "model_accuracy": 95.5
}
```

---

### 4. Get My Models (Authenticated)

Get all models belonging to the authenticated user.

**Endpoint:** `GET /api/models/my/`  
**Authentication:** Required  
**Permissions:** Authenticated users only

#### Request Example:
```bash
curl -u username:password http://127.0.0.1:8000/api/models/my/
```

#### Response Example:
```json
{
  "models": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "My First Model",
      "description": "Testing the platform",
      "status": "completed",
      "accuracy": 92.0,
      "val_accuracy": 89.5,
      "is_public": false,
      "created_at": "2025-11-03T10:30:00Z"
    },
    {
      "id": "660e8400-e29b-41d4-a716-446655440001",
      "name": "Experimental Model",
      "description": "High epoch training",
      "status": "training",
      "accuracy": null,
      "val_accuracy": null,
      "is_public": false,
      "created_at": "2025-11-03T14:20:00Z"
    }
  ],
  "count": 2
}
```

---

## Error Responses

All endpoints return standard HTTP status codes and JSON error messages.

### Common Error Codes:

#### 400 Bad Request
```json
{
  "error": "No image provided"
}
```

#### 403 Forbidden
```json
{
  "error": "Model is private"
}
```

#### 404 Not Found
```json
{
  "error": "Model not found"
}
```

#### 500 Internal Server Error
```json
{
  "error": "Error processing image: Invalid image format"
}
```

---

## Complete Usage Examples

### Python Example

```python
import requests
import base64
from pathlib import Path

class CatDogAPI:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def login(self, username, password):
        """Login using basic auth"""
        self.session.auth = (username, password)
    
    def list_public_models(self):
        """Get all public models"""
        response = self.session.get(f"{self.base_url}/api/models/")
        return response.json()
    
    def get_model_details(self, model_id):
        """Get model details"""
        response = self.session.get(f"{self.base_url}/api/model/{model_id}/")
        return response.json()
    
    def predict_image(self, model_id, image_path):
        """Predict image class"""
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Make prediction
        response = self.session.post(
            f"{self.base_url}/api/model/{model_id}/predict/",
            json={'image': image_data}
        )
        return response.json()
    
    def my_models(self):
        """Get my models (requires authentication)"""
        response = self.session.get(f"{self.base_url}/api/models/my/")
        return response.json()

# Usage
api = CatDogAPI()

# List public models
models = api.list_public_models()
print(f"Found {models['count']} public models")

# Use a model to classify an image
if models['count'] > 0:
    model_id = models['models'][0]['id']
    result = api.predict_image(model_id, 'test_cat.jpg')
    print(f"Prediction: {result['result']} ({result['confidence']:.2f}% confidence)")

# Login and get my models
api.login('username', 'password')
my_models = api.my_models()
print(f"You have {my_models['count']} models")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');
const fs = require('fs');

class CatDogAPI {
  constructor(baseURL = 'http://127.0.0.1:8000') {
    this.client = axios.create({ baseURL });
  }

  setAuth(username, password) {
    this.client.defaults.auth = { username, password };
  }

  async listPublicModels() {
    const response = await this.client.get('/api/models/');
    return response.data;
  }

  async getModelDetails(modelId) {
    const response = await this.client.get(`/api/model/${modelId}/`);
    return response.data;
  }

  async predictImage(modelId, imagePath) {
    // Read and encode image
    const imageBuffer = fs.readFileSync(imagePath);
    const imageBase64 = imageBuffer.toString('base64');

    // Make prediction
    const response = await this.client.post(
      `/api/model/${modelId}/predict/`,
      { image: imageBase64 }
    );
    return response.data;
  }

  async myModels() {
    const response = await this.client.get('/api/models/my/');
    return response.data;
  }
}

// Usage
(async () => {
  const api = new CatDogAPI();

  // List public models
  const models = await api.listPublicModels();
  console.log(`Found ${models.count} public models`);

  // Use a model
  if (models.count > 0) {
    const modelId = models.models[0].id;
    const result = await api.predictImage(modelId, 'test_cat.jpg');
    console.log(`Prediction: ${result.result} (${result.confidence.toFixed(2)}% confidence)`);
  }

  // Login and get my models
  api.setAuth('username', 'password');
  const myModels = await api.myModels();
  console.log(`You have ${myModels.count} models`);
})();
```

---

## Rate Limiting

Currently, there are no rate limits on API endpoints. In production, consider implementing rate limiting using Django Rest Framework throttling.

## CORS

For cross-origin requests from web applications, you may need to configure CORS. Install `django-cors-headers`:

```bash
pip install django-cors-headers
```

Add to `settings.py`:
```python
INSTALLED_APPS = [
    ...
    'corsheaders',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    ...
]

CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",  # Your frontend URL
]
```

---

## Testing API with Postman

Import this collection into Postman:

**Collection Variables:**
- `base_url`: `http://127.0.0.1:8000`
- `model_id`: Your model UUID

**Requests:**
1. GET List Models: `{{base_url}}/api/models/`
2. GET Model Details: `{{base_url}}/api/model/{{model_id}}/`
3. POST Predict: `{{base_url}}/api/model/{{model_id}}/predict/`
   - Body (JSON): `{"image": "base64_string_here"}`

---

## Support

For issues or questions:
- Check the Django debug page when `DEBUG=True`
- Review logs in the terminal
- Ensure TensorFlow is properly installed
- Verify model files exist in the media directory