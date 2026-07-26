# AI Image Classifier Dashboard

A python full-stack deep learning web application that classifies images into 1,000 object categories using a pre-trained **ResNet-50** Neural Network trained on the ImageNet dataset.

---

## Technologies & Tools Used

### **Backend & Deep Learning**
- **Python 3.12**: Core programming language.
- **PyTorch (`torch`)**: Deep learning framework for inference.
- **Torchvision (`torchvision`)**: Official computer vision library providing pre-trained ResNet-50 weights and image transformations.
- **Flask**: Lightweight WSGI web application framework for handling HTTP requests, file uploads, and template rendering.
- **Pillow (`PIL`)**: Image loading and processing library.
- **Requests**: HTTP client for fetching ImageNet class labels dynamically.

### **Frontend & User Interface**
- **HTML5**: Semantic web application layout with a responsive two-column grid.
- **CSS3 (Vanilla)**: Custom dark theme styling with black background (`#000000`), deep navy card containers (`#0a0f1d`), bluish-white typography (`#e2e8f0`), and cyan-blue gradient accents (`#38bdf8`).
- **Google Fonts**: Modern *Plus Jakarta Sans* font family for clean readability.
- **Interactive JavaScript**: Client-side drag-and-drop file upload detection and real-time filename preview.

---

## Key Features

1. **Top Match & Top-5 Probabilities**: Computes exact softmax confidence percentages and top 5 predictions for any uploaded image.
2. **Standard Preprocessing**: Uses PyTorch's official `ResNet50_Weights.DEFAULT.transforms()` pipeline (Resizing to 256px, Center Crop to 224px, Tensor conversion, and ImageNet mean/std normalization).
3. **Clean Modern Dashboard**: Responsive dashboard layout featuring file preview, prediction stats, and visual probability progress bars.
4. **Command Line & Web Support**: Run as a Flask web dashboard or directly from the CLI.

---

## Project Structure

```
image_classifier/
│
├── app.py                  # Command-line interface (CLI) for running single image classification
├── requirements.txt        # Python package dependencies
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
│   └── resnet.py           # ResNet-50 model initialization function
│
└── utils/
    └── preprocess.py       # Image preprocessing module using ResNet50 weights transforms
```

---

## How to Run

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Run Web Dashboard**
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
