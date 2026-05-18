# AI-Driven Fashion Recommendation System using YOLOv8 and FashionCLIP

## Overview

This project is an AI-based fashion recommendation system focused on personal wardrobe management. The system detects garments from uploaded wardrobe images, extracts visual and semantic features, and generates event-based outfit recommendations using a hybrid scoring approach.

Unlike many existing fashion recommendation systems that focus mainly on e-commerce, this project is designed to help users organise and utilise their own clothing collections more effectively.

---

## Features

* Garment detection using YOLOv8n
* Automatic garment cropping using detected bounding boxes
* Category mapping to user-friendly fashion labels
* Colour detection using HSV colour space and K-means clustering
* Style representation using FashionCLIP embeddings
* Hybrid outfit scoring system
* Event-based outfit recommendation generation
* Interactive Streamlit web application

---

## Technologies Used

* Python
* YOLOv8n
* FashionCLIP
* PyTorch
* OpenCV
* Streamlit
* NumPy
* Scikit-learn
* Pillow

---

## System Pipeline

1. User uploads wardrobe images
2. YOLOv8 detects garment regions
3. Garments are cropped using bounding box coordinates
4. Detected classes are mapped into fashion categories
5. Features are extracted:

   * Colour features
   * FashionCLIP embeddings
6. Clothing items are stored in wardrobe database
7. Outfit combinations are generated
8. Hybrid scoring ranks outfits
9. Final recommendations are displayed

---

## Dataset

The project uses the DeepFashion2 dataset for garment detection training.

### Training Subset

* Training Images: 5000
* Validation Images: 800

---

## Model Training

### YOLOv8n Training Configuration

* Epochs: 30
* Image Size: 512 × 512
* Model: YOLOv8n
* Framework: Ultralytics YOLO

---

## Hybrid Scoring System

The recommendation engine combines multiple scoring factors:

* Category validity
* Event suitability
* Formality alignment
* Colour harmony
* Style compatibility
* YOLO label consistency

The final outfit score is calculated using weighted scoring.

---

## Results

### System Performance

| Component                      | Result |
| ------------------------------ | ------ |
| Garment Detection Precision    | 0.72   |
| mAP@0.5                        | 0.596  |
| Category Prediction Accuracy   | 93.3%  |
| Colour Detection Accuracy      | 66.7%  |
| Outfit Recommendation Accuracy | 86.7%  |
| System Success Rate            | 100%   |

---

## Project Structure

```text
FASHION_AI_PROJECT/
│
├── assets/
├── data/
├── models/
├── notebooks/
├── src/
│   ├── detection/
│   ├── recommendation/
│   ├── scoring.py
│   └── feature_extraction/
├── app.py
├── requirements.txt
└── README.md
```

---

## Future Improvements

* Use segmentation models for more accurate garment extraction
* Improve colour detection using deep learning methods
* Add personalised recommendation learning
* Expand training dataset
* Deploy as a cloud-based web application

---

## Author

**Waruni Sawindi Liyanapathirana**
BSc (Hons) Computer Science (Artificial Intelligence)
University of Hertfordshire

---

## Research Context

This project was developed as a final year undergraduate research project exploring the integration of:

* Computer Vision
* Deep Learning
* Recommendation Systems
* Semantic Embedding Models
* Context-Aware Fashion Recommendation
