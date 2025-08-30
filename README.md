# 🍎🍌 Rotten Fruit – Freshness Detection using Machine Learning

Rotten Fruit is a **machine learning project** that predicts the **freshness stage of fruits** (Raw, Fresh, Spoiled) using multiple sensors (Near Infrared, RGB color, temperature, humidity, and VOC).  
The project demonstrates the **full ML lifecycle**: data collection, preprocessing, model training, evaluation, and deployment with a beautiful **Streamlit app**.

---

## 🚀 Key Features
- Detects fruit freshness stage across **3 classes**:  
  - **Raw** 🥭  
  - **Fresh** 🍏  
  - **Spoiled** 🍂  

- Uses **multi-modal sensor data**:  
  - **NIR 850 & 940 nm** (near infrared reflectance)  
  - **RGB channels** (R, G, B color intensities)  
  - **Temperature (°C)**  
  - **Humidity (%)**  
  - **VOC gas concentration (ppm)**  

- Interactive **Streamlit web app**:  
  - 📊 View evaluation metrics (Accuracy, Precision, Recall, F1, ROC, PR curves)  
  - 🔬 Sensor explanations with images  
  - 🔮 Test predictions manually by entering sensor values  
  - 🎨 Stylish **glassmorphic UI** with transparent sidebar & background image  

---

## 📂 Project Structure
```
rotten-fruit/
│
├── fruit-modrl/ # Pretrained model + encoders + scaler
│ ├── model.pkl # Trained Logistic Regression model
│ ├── scaler.pkl # Scaler used during preprocessing
│ ├── fruit_encoder.pkl # Label encoder for fruit type
│ ├── stage_encoder.pkl # Label encoder for freshness stage
│ ├── fruits_data.csv # Dataset used for evaluation
│ └── main_background.jpg # Background image for app
│
├── sensors/ # Sensor images
│ ├── nir-all.png
│ ├── R G B.jpg
│ ├── Temp.jpg
│ ├── dh22.jpg
│ └── voc.jpg
│
├── Rotten_fruit.py # Main Streamlit app
├── requirements.txt # Python dependencies
└── README.md # Project description
└── logo.png # logo image for app
```
---

## 📊 Workflow

1. **Data Collection**  
   Fruit sensor data was collected for multiple fruits (e.g., mango, apple, watermelon) with freshness stage labels (`raw`, `fresh`, `spoiled`).  

2. **Preprocessing**  
   - Encoded categorical values (fruit name, freshness stage).  
   - Normalized numeric features using `StandardScaler`.  
   - Split into training/testing sets.  

3. **Exploratory Data Analysis (EDA)**  
   - Visualized distributions of sensor readings.  
   - Analyzed correlations between NIR/RGB and freshness.  
   - Detected outliers and missing values.  

4. **Feature Engineering**  
   - Combined NIR (850 & 940) features.  
   - Grouped RGB channels for visual maturity.  
   - Engineered VOC thresholds for spoilage.  

5. **Model Training**  
   - Trained a **Logistic Regression model** on 8 sensor features.  
   - Compared multiple algorithms before selecting Logistic Regression for balance of performance & interpretability.  

6. **Evaluation**  
   - Metrics: Accuracy, Precision, Recall, F1-score (multi-class weighted).  
   - Plots: Confusion Matrix, ROC curves, Precision–Recall curves.  

7. **Deployment (Streamlit)**  
   - Built interactive UI with **tabs**:  
     - **Overview**: Project description  
     - **Sensors**: Explanations + images  
     - **Model Evaluation**: Metrics + plots from dataset  
     - **Test the Project**: Enter sensor values & predict freshness  

---

## 🖼 Streamlit App UI

- **Overview Tab** – Project introduction & summary.  
- **Sensors Tab** – Visual explanation of each sensor type (NIR, RGB, Temp, Humidity, VOC).  
- **Model Evaluation Tab** – Preview dataset, performance metrics, confusion matrix, ROC, and PR curves.  
- **Test Tab** – Interactive input fields to predict fruit freshness in real time.  

---

## ⚙️ Installation & Running Locally

1. **Clone this repository**:
   ```
   bash
   git clone https://github.com/your-username/rotten-fruit.git
   cd rotten-fruit
   ```
2. **Create a virtual environment**:
```
python -m venv .venv
source .venv/bin/activate   # (Linux/Mac)
.venv\Scripts\activate      # (Windows)
```
3. **Install dependencies**:
```pip install -r requirements.txt```

4. **Run the Streamlit app**:
```streamlit run Rotten_fruit.py```

# 🌐 Deployment
The project is deployed using Streamlit Cloud:
👉 rotten-fruit.streamlit.app

# 📦 Requirements
See requirements.txt
Main dependencies:
```
numpy

pandas

scikit-learn

matplotlib

streamlit
```
# 👨‍💻 Authors
Faisal Alfodaily – Software Engineer & AI Enthusiast
King Saud University
