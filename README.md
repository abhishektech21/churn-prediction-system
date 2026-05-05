# 🚀 Cloud-Based Customer Churn Prediction System

A complete Machine Learning project that predicts whether a customer will churn or not.
The system is built using **FastAPI**, containerized with **Docker**, and deployed on cloud.

---

## 📌 Project Overview

This project:

* Trains a machine learning model on customer data
* Exposes predictions via a REST API
* Runs inside a Docker container
* Can be deployed on cloud platforms like Render / AWS

---

## 🧠 Architecture

```
User → FastAPI API → ML Model → Prediction → Response
```

---

## 📁 Project Structure

```
churn-prediction-system/
│── app.py              # FastAPI application
│── train.py            # Model training script
│── model.pkl           # Trained ML model
│── requirements.txt    # Dependencies
│── Dockerfile          # Docker configuration
│── .dockerignore
```

---

## ⚙️ Installation & Setup (Local)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/churn-prediction-system.git
cd churn-prediction-system
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Train the model

```bash
python train.py
```

👉 This will generate:

```
model.pkl
```

---

### 4️⃣ Run the API

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

---

### 5️⃣ Open API Docs

```
http://127.0.0.1:8000/docs
```

---

## 🧪 API Usage

### 🔹 Endpoint

```
POST /predict
```

### 🔹 Sample Request

```json
{
  "tenure": 2,
  "monthly_charges": 110.0,
  "total_charges": 220.0,
  "contract": "Month-to-month",
  "internet_service": "Fiber optic",
  "gender": "Female"
}
```

### 🔹 Sample Response

```json
{
  "prediction": 1,
  "label": "Churn"
}
```

---

## 🐳 Docker Setup

### Build Docker Image

```bash
docker build -t churn-app .
```

---

### Run Container

```bash
docker run -p 8000:8000 churn-app
```

---

### Access API

```
http://localhost:8000/docs
```

---

## ☁️ Cloud Deployment (Render)

### Steps:

1. Push code to GitHub
2. Go to https://render.com
3. Create **New Web Service**
4. Connect your repository
5. Select **Docker Environment**
6. Click **Deploy**

---

### Access Deployed API

```
https://your-render-url/docs
```

---

## 📊 Features Used

* tenure
* monthly_charges
* total_charges
* contract
* internet_service
* gender

---

## 🛠️ Technologies Used

* Python
* FastAPI
* Scikit-learn
* Docker
* Render (Cloud)

---

## 🎯 Project Outcome

* Built an end-to-end ML system
* Exposed model via REST API
* Deployed on cloud
* Demonstrated real-world MLOps workflow

---

## 🎤 Viva Explanation (Short)

This project predicts customer churn using a machine learning model.
The model is trained using Python and deployed using FastAPI.
It is containerized using Docker and hosted on a cloud platform.

---

## 👨‍💻 Author

Your Name
GitHub: https://github.com/YOUR_USERNAME
