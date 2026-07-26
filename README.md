# AI Image Classifier Dashboard

A python full-stack deep learning web application that classifies images into 1,000 object categories using a pre-trained **ResNet-18** Neural Network trained on the ImageNet dataset.

🌐 **Live Demo**: [https://image-classifier-vub2.onrender.com](https://image-classifier-vub2.onrender.com)

---

## Technologies & Tools Used

### **Backend & Deep Learning**
- **Python 3.12**: Core programming language.
- **PyTorch (`torch`)**: Lightweight CPU-optimized deep learning framework for inference.
- **Torchvision (`torchvision`)**: Official computer vision library providing pre-trained ResNet-18 weights and image transformations.
- **Flask**: WSGI web application framework handling HTTP requests, file uploads, and template rendering.
- **Gunicorn**: Production WSGI HTTP server configured with single-worker execution for cloud hosting.
- **Pillow (`PIL`)**: Image loading and pre-processing library.
- **Requests**: HTTP client fetching ImageNet class labels dynamically.

### **Frontend & User Interface**
- **HTML5**: Semantic layout with responsive two-column grid.
- **CSS3 (Vanilla)**: Custom dark theme with black background (`#000000`), deep navy cards (`#0a0f1d`), bluish-white typography (`#e2e8f0`), and cyan-blue gradient accents (`#38bdf8`).
- **Google Fonts**: *Plus Jakarta Sans* font family.
- **Interactive JavaScript**: Drag-and-drop file upload zone and real-time filename preview.

---

## Key Features

1. **Top Match & Top-5 Probabilities**: Computes exact softmax confidence percentages and top 5 predictions for any uploaded image.
2. **Memory-Optimized Model**: ResNet-18 architecture tuned for cloud deployment under 512MB RAM constraints.
3. **Standard Preprocessing**: Uses PyTorch's official `ResNet18_Weights.DEFAULT.transforms()` pipeline (Resizing to 256px, Center Crop to 224px, Tensor conversion, and ImageNet normalization).
4. **Clean Modern Dashboard**: Responsive dashboard layout featuring file preview, prediction stats, and visual probability progress bars.
5. **Cloud Deployed**: Live production deployment hosted on Render.

---

## Project Structure

```
image_classifier/
│
├── app.py                  # Command-line interface (CLI) for running single image classification
├── requirements.txt        # Python package dependencies (CPU PyTorch wheels)
├── Procfile                # Gunicorn process configuration for cloud deployment
├── README.md               # Project documentation
│
├── backend/
│   └── server.py           # Flask server routing & model inference integration
│
├── frontend/
│   ├── static/
│   │   └── style.css       # Custom black & bluish-white responsive CSS stylesheet
│   └── templates/
│       └── index.html      # Jinja2 HTML dashboard template
│
├── models/
│   └── resnet.py           # ResNet-18 model initialization function
│
└── utils/
    └── preprocess.py       # Image preprocessing module using ResNet18 weights transforms
```

---

## How to Run

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Run Web Dashboard Locally**
Run the Flask server from the root directory:
```bash
python backend/server.py
```
Open your browser and navigate to: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

### **3. Run CLI Classifier**
Run direct image inference via command line:
```bash
python app.py --image path/to/your/image.jpg
```

---

## Author
Developed by **Jaishree Verma**
