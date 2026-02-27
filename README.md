# 🩺 X-Ray Lung Classifier (Pneumonia Detection)

## 📌 Problem Statement

Pneumonia is an inflammatory condition of the lungs affecting primarily the small air sacs (alveoli). It can be life-threatening if not diagnosed early.

Manual diagnosis using chest X-rays requires expert radiologists and can sometimes be time-consuming. In many regions, medical facilities lack sufficient specialists, which delays diagnosis.

This project aims to build an AI-based deep learning system that can classify chest X-ray images into:

- ✅ NORMAL
- ❌ PNEUMONIA

The goal is to assist medical professionals by providing fast and automated screening support.

---

## 🚀 Solution Approach

We developed an end-to-end Deep Learning pipeline that includes:

1. Data Ingestion (from AWS S3)
2. Data Transformation & Augmentation
3. Model Training (Custom CNN Architecture)
4. Model Evaluation
5. Model Pusher (Upload best model to S3)
6. FastAPI-based deployment
7. Docker containerization

The trained model is deployed as an API that allows users to upload X-ray images and receive predictions.

---

## 📊 Dataset Used

- Publicly available chest X-ray dataset
- Binary classification:
  - NORMAL
  - PNEUMONIA
- Dataset structure:


chest_xray/
├── train/
│ ├── NORMAL/
│ └── PNEUMONIA/
├── test/
│ ├── NORMAL/
│ └── PNEUMONIA/



---

## 🧠 Model Used

- Custom CNN Architecture
- Multiple Convolution Blocks
- Batch Normalization
- ReLU Activation
- Max Pooling
- Global Average Pooling
- LogSoftmax Output Layer

---

## 🏗 Project Architecture

xray/
│
├── components/
│ ├── data_ingestion
│ ├── data_transformation
│ ├── model_training
│ ├── model_evaluation
│ ├── model_pusher
│
├── exception/
├── logger/
├── utils/
│
├── app.py
├── train.py
├── requirements.txt



---

## ⚙ Tech Stack Used

- Python
- PyTorch
- FastAPI
- AWS (S3, EC2)
- Docker
- GitHub Actions (CI/CD)

---

## ☁ Infrastructure Required

- AWS S3 (Dataset & Model Storage)
- AWS EC2 (Deployment)
- Docker
- GitHub

---

## 🛠 How To Run (Local Setup)

### 1️⃣ Clone Repository

```bash
git clone <your_repo_url>
cd <project_folder>


2️⃣ Create Virtual Environment

Using Conda:

conda create -n xray python=3.9
conda activate xray

OR using venv:

python -m venv venv
venv\Scripts\activate


3️⃣ Install Requirements
pip install -r requirements.txt



4️⃣ Set AWS Environment Variables (Windows PowerShell)
$env:AWS_ACCESS_KEY_ID="your_access_key"
$env:AWS_SECRET_ACCESS_KEY="your_secret_key"
$env:AWS_DEFAULT_REGION="ap-south-1"



5️⃣ Train Model
python train.py
6️⃣ Run API Server
uvicorn app:app --reload

Open in browser:

http://127.0.0.1:8000/docs  



🎯 Conclusion

This project demonstrates an end-to-end Deep Learning + MLOps workflow for medical image classification.

It covers:

**
Model Development

Modular ML Pipeline

Cloud Storage Integration

API Deployment

Containerization

Production Readiness


**


The system can assist doctors in faster screening and decision-making.