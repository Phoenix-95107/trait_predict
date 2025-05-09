# Trait Predict

A facial analysis system that predicts personality traits and characteristics from facial images using computer vision and machine learning techniques.

## Overview

Trait Predict analyzes facial expressions, micro-expressions, and facial features to provide insights into various personality dimensions. The system processes images through multiple specialized analyzers to extract different trait categories:

1. **Relationship & Empathy** - Analyzes trust, openness, empathy, and conflict avoidance tendencies
2. **Work DNA & Focus** - Evaluates persistence, focus, structure preference, and risk tolerance in work settings
3. **Creativity Pulse** - Measures ideation capability, openness, originality, and attention to detail
4. **Stress Resilience** - Assesses stress indicators, emotional regulation, and overall resilience
5. **Adventure & Exploration** - Evaluates openness to experience, novelty seeking, risk tolerance, and planning preferences
6. **Learning & Growth** - Measures intellectual curiosity, reflective tendency, and structured learning preferences

Additionally, the system provides OCEAN personality model predictions (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism).

## Technical Architecture

The system consists of:

- **FastAPI Backend** - Handles image uploads and coordinates analysis
- **Computer Vision Pipeline** - Uses MediaPipe and OpenCV for facial landmark detection and analysis
- **Neural Network Model** - A CNN-based classifier for OCEAN personality traits
- **Specialized Analyzers** - Six different analyzers for specific trait categories

## Key Features

- Multi-image analysis for more accurate predictions
- Parallel processing for improved performance
- Comprehensive trait analysis across multiple dimensions
- Facial Action Unit (AU) detection for micro-expression analysis
- Eye gaze and iris analysis for additional insights

## Technologies Used

- Python
- FastAPI
- PyTorch
- MediaPipe
- OpenCV
- NumPy
- Py-Feat (for Action Unit detection)
- OpenAI API integration

## API Endpoints

### `/freemium_analyze`

Analyzes uploaded images and returns comprehensive trait predictions.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: List of image files

**Response:**
- JSON object containing trait predictions across all six categories

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Phoenix-95107/trait_predict.git
cd trait_predict
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
echo "OPENAI_API_KEY=your_openai_api_key" > .env
```

4. Run the application:
```bash
uvicorn app:app --reload
```

## Usage

Send a POST request to the `/freemium_analyze` endpoint with image files to analyze facial traits:

```bash
curl -X POST "http://localhost:8000/freemium_analyze" 
  -H "accept: application/json" 
  -H "Content-Type: multipart/form-data" 
  -F "files=@photo1.jpg"  
  -F "files=@photo2.jpg"  
  -F "files=@photo3.jpg"  
  -F "files=@photo4.jpg"  
  -F "files=@photo5.jpg"

```