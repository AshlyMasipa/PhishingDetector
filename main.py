from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import onnxruntime as ort  # Changed from onnxruntime_web
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="Phishing Detection API", version="1.0.0")

# Add CORS middleware to allow requests from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
word_vectorizer = None
char_vectorizer = None
onnx_session = None

def load_models():
    """Load ML models with error handling"""
    global word_vectorizer, char_vectorizer, onnx_session
    
    try:
        # Load vectorizers
        word_vectorizer = joblib.load('model_files/word_vectorizer.pkl')
        char_vectorizer = joblib.load('model_files/char_vectorizer.pkl')
        
        # Load ONNX model - fixed file extension
        onnx_session = ort.InferenceSession('model_files/phishing_model.onnx')
        
        print("✅ All models loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

# Load models at startup
models_loaded = load_models()

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    is_phishing: bool
    status: str = "success"

def extract_features(text):
    """Extract features from text matching your training pipeline"""
    if word_vectorizer is None or char_vectorizer is None:
        raise ValueError("Models not loaded properly")
    
    # Basic text cleaning
    text = text.lower().strip()
    
    # Handcrafted features
    text_length = len(text)
    special_char_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
    
    # Vectorize using your trained vectorizers
    word_features = word_vectorizer.transform([text])
    char_features = char_vectorizer.transform([text])
    
    # Combine features
    combined_features = np.hstack([
        word_features.toarray(),
        char_features.toarray(),
        [[text_length, special_char_count]]
    ])
    
    return combined_features

@app.post("/predict", response_model=PredictionResponse)
async def predict_phishing(request: PredictionRequest):
    try:
        if not models_loaded:
            return PredictionResponse(
                prediction=0,
                confidence=0.0,
                is_phishing=False,
                status="error: Models not loaded"
            )
        
        # Extract features
        features = extract_features(request.text)
        
        # Make prediction using ONNX model
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        
        prediction = onnx_session.run(
            [output_name], 
            {input_name: features.astype(np.float32)}
        )[0]
        
        # Get probability of class 1 (phishing)
        probability = float(prediction[0][1])
        is_phishing = probability > 0.5
        
        return PredictionResponse(
            prediction=int(is_phishing),
            confidence=round(probability, 4),
            is_phishing=is_phishing
        )
        
    except Exception as e:
        return PredictionResponse(
            prediction=0,
            confidence=0.0,
            is_phishing=False,
            status=f"error: {str(e)}"
        )

@app.get("/")
async def root():
    return {
        "message": "Phishing Detection API", 
        "status": "active",
        "models_loaded": models_loaded,
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    try:
        if models_loaded:
            # Test with a simple prediction
            test_features = extract_features("test message")
            return {
                "status": "healthy", 
                "models_loaded": True,
                "service": "phishing-detection-api"
            }
        else:
            return {
                "status": "unhealthy", 
                "models_loaded": False,
                "error": "Models failed to load"
            }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)