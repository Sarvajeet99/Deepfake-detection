# Deepfake Detection using Xception

This project implements a deepfake detection model using the **Xception architecture**, designed to identify subtle facial manipulation artifacts in images.

---

## 📌 Features

* Xception-based deep learning model (pretrained on ImageNet)
* Input resolution: **299×299**
* Binary classification (Real vs Fake)
* Modular code structure (model, dataset, training)
* GPU support (recommended)

---

## 🧠 Model Architecture

* Backbone: Xception (via timm)
* Classifier: Fully connected layers with dropout
* Loss Function: Binary Cross Entropy
* Optimizer: Adam

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/deepfake-xception.git
cd deepfake-xception
```

---

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

* Windows:

```bash
venv\Scripts\activate
```

* Mac/Linux:

```bash
source venv/bin/activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

You can use publicly available datasets such as:

* FaceForensics++
* DFDC (DeepFake Detection Challenge)

### 🔗 Kaggle Dataset (Recommended)

https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images

---

## ⬇️ Download Dataset using Kaggle API

### Step 1: Install Kaggle

```bash
pip install kaggle
```

---

### Step 2: Add Kaggle API key

1. Go to Kaggle → Account → Create API Token
2. Download `kaggle.json`

Place it in:

```bash
~/.kaggle/kaggle.json
```

For Windows:

```bash
C:\Users\YourUsername\.kaggle\kaggle.json
```

---

### Step 3: Download dataset

```bash
kaggle datasets download -d xhlulu/140k-real-and-fake-faces
unzip 140k-real-and-fake-faces.zip
```

---

## 📁 Dataset Structure

Organize your dataset in the following format:

```bash
dataset/
│
├── real/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
│
└── fake/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

⚠️ Ensure:

* All real images are inside `real/`
* All fake images are inside `fake/`

---

## ⚙️ Training

Run the training script:

```bash
python train.py
```

🔍 Inference (Testing on a Single Image)

After training, you can test the model on a new image using:

python infer.py image.jpg
📝 Expected Output

The model will output the probability of the image being real or fake, for example:

Fake: 87.23%
Real: 12.77%

---

## 📈 Notes

* Recommended input size: **299×299**
* Use a GPU for faster training
* Ensure dataset is balanced (equal real and fake samples)

---

## 🚀 Future Improvements

* Video-based detection (temporal models)
* Attention mechanisms
* Better generalization with augmentation

---

## 📜 License

This project is intended for educational and research purposes only.


