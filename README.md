# AI-Powered Signature Verification System

A production-ready signature verification web application using advanced machine learning and computer vision techniques.

## Features

- **Real AI/ML Implementation**: Uses Siamese Neural Networks for signature similarity learning
- **Advanced Preprocessing**: Multi-stage image processing with OpenCV for optimal feature extraction
- **High Accuracy**: Target ≥90% accuracy in distinguishing genuine vs forged signatures
- **Data Augmentation**: Intelligent augmentation to improve model robustness
- **Real-time Training**: Stream training progress with live updates
- **Professional UI**: Clean, responsive interface with confidence scoring

## Architecture

### Frontend (React + TypeScript)
- Single-page application with dual training/verification interface
- Real-time progress updates during model training
- Drag-and-drop file uploads with image preview
- Confidence score visualization with progress bars

### Backend (FastAPI + Python)
- RESTful API with streaming responses for training progress
- Advanced computer vision preprocessing pipeline
- Siamese Neural Network implementation with TensorFlow/Keras
- Robust error handling and logging

### AI/ML Pipeline
1. **Image Preprocessing**: Grayscale conversion, noise reduction, adaptive thresholding, morphological operations
2. **Feature Extraction**: CNN-based deep learning embeddings
3. **Model Training**: Siamese network with contrastive learning
4. **Verification**: Embedding similarity comparison with confidence scoring

## Quick Start

### Prerequisites
- Node.js (v16+)
- Python (3.8+)
- pip

### Installation

1. Start the frontend:
```bash
npm install
npm run dev
```

2. Start the backend (in a new terminal):
```bash
.venv/Scripts/activate
cd backend
pip install -r requirements.txt
python main.py
```

The application will be available at `http://localhost:8000`

### Usage

1. **Training Phase**:
   - Upload at least 3 signature images of the same person
   - Click "Train AI Model" to start the training process
   - Wait for training completion (typically 2-5 minutes)

2. **Verification Phase**:
   - Upload a signature image to verify
   - Click "Verify Signature" to check against trained model
   - View results with confidence score

## Technical Details

### Image Processing Pipeline
- Bilateral filtering for noise reduction
- Adaptive thresholding for edge enhancement
- Morphological operations for cleanup
- Smart cropping to signature bounds
- Aspect ratio preservation during resize
- Pixel normalization for ML compatibility

### Neural Network Architecture
- Base CNN: 4 convolutional blocks with batch normalization
- Global average pooling for translation invariance
- Dense layers with dropout for regularization
- 128-dimensional embedding space
- L1 distance-based similarity computation

### Data Augmentation
- Random rotation (-10° to +10°)
- Random scaling (0.9x to 1.1x)
- Gaussian noise injection
- Maintains signature characteristics while increasing diversity

### Performance Optimizations
- Batch processing for efficient training
- Streaming responses for real-time updates
- Optimized image preprocessing pipeline
- Memory-efficient embedding storage

## API Endpoints

- `POST /train` - Train model with signature images
- `POST /verify` - Verify signature against trained model
- `GET /model_status` - Check training status

## Dependencies

### Python Backend
- FastAPI - Modern web framework
- TensorFlow/Keras - Deep learning framework
- OpenCV - Computer vision library
- NumPy - Numerical computing
- Pillow - Image processing

### Frontend
- React 18 - UI framework
- TypeScript - Type safety
- Tailwind CSS - Styling
- Lucide React - Icons

## Security Considerations
- Input validation and sanitization
- Error handling to prevent information leakage
- CORS configuration for development
- File type and size restrictions

## Future Enhancements
- Database integration for signature storage
- Multi-user support with authentication
- Batch verification capabilities
- Advanced analytics dashboard
- Mobile app integration
- Cloud deployment options