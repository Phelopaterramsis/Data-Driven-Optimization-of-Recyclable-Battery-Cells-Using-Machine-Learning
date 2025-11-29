# 🔋 Data-Driven Optimization of Recyclable Battery Cells Using Machine Learning

This folder contains my first research paper, where we designed a custom battery-tester circuit and built machine-learning models to estimate **State of Charge (SoC)**, **State of Health (SoH)**, and **Remaining Useful Life (RUL)** for recyclable battery cells used in electrified systems.

## 📌 Paper Details

- Title: **Data-Driven Optimization of Recyclable Battery Cells for Electrified Systems Using Machine Learning Regression Models**  
- Authors: Ahmed Hereiz, **Phelopater Ramsis**, Youssef M. Gunaidi, Mennatallah A. Gomaa, Doha A. Gomaa, Ali Shoma, Basem M. Badr   
- Status: Research paper manuscript  
:contentReference[oaicite:1]{index=1}

---

## 🧾 Abstract Overview

The paper presents:

- A **custom-designed battery tester circuit** to measure voltage, current, temperature, internal resistance, and capacity  
- A supervised ML pipeline to estimate **SoC**, **SoH**, and predict **RUL**  
- Comparison of multiple regression models  
- A hybrid numerical method combining **capacity fade** + **resistance growth** for SoH estimation  
- Evidence that Random Forest and Gradient Boosting provide the **lowest MAE** and best generalization  
:contentReference[oaicite:2]{index=2}

---

## 🛠️ 1. Hardware System – Upgraded Battery Tester

The battery-tester system includes:

- **Arduino Nano microcontroller**  
- **LM358 amplifier**  
- **Temperature sensor**  
- **3× 100Ω power resistors** for constant discharge  
- **LCD (I²C)**  
- **Improved firmware** with better menus, alerts, PWM current control, and multi-page monitoring  

Key improvements (pages 4–5):

- Real-time SoC verification using **open-circuit voltage + internal resistance lookup table**  
- 9 selectable discharge currents  
- Temperature monitoring for overheating protection  
- Faster data-logging using UART  
:contentReference[oaicite:3]{index=3}

---

## 📊 2. Dataset Description

Data was collected from:

- **Good recyclable cells**  
- **Bad/aged recyclable cells**

Features measured:

- Voltage  
- Current  
- Internal resistance  
- Temperature  
- Capacity  
- SoC (from CC + FW validation)

Two experiment modes:

- With temperature  
- Without temperature  

Temperature significantly improved model accuracy, especially for bad cells.  
:contentReference[oaicite:4]{index=4}

---

## 🤖 3. Machine Learning Models

Models tested for **SoC estimation**:

- Random Forest  
- Gradient Boosting  
- AdaBoost  
- Linear, Ridge, Lasso Regression  
- SVR  
- Decision Tree  
- KNN  

ML models trained using:

- **80% training / 20% testing split**  
- **Independent unseen dataset** for second verification  
:contentReference[oaicite:5]{index=5}

### Best Models

- **Random Forest** → lowest MAE (with & without temperature)  
- **Gradient Boosting** → competitive accuracy and strong generalization  
- **Decision Tree** → lowest computation time  
:contentReference[oaicite:6]{index=6}

---

## 💡 4. State of Health (SoH) Estimation

A hybrid numerical method was introduced:

- **Capacity-based SoH:**  
  \[
  \text{SoH}_\text{cap}(t) = \frac{Q(t)}{Q_{rated}}
  \]

- **Resistance-based SoH:**  
  \[
  \text{SoH}_\text{res}(t) = \frac{R(0)}{R(t)}
  \]

- **Final SoH:**  
  \[
  \text{SoH}_{final} = \frac{1}{2} (\text{SoH}_\text{cap} + \text{SoH}_\text{res})
  \]

Smoothed with a 5-point moving average filter.  
:contentReference[oaicite:7]{index=7}

---

## 🔮 5. Remaining Useful Life (RUL) Prediction

RUL prediction used Kaggle dataset features such as:

- Cycle index  
- Voltages  
- Time constants  
- Discharge intervals  
- Charge durations  

Evaluated ML models:

- Random Forest → **best MAE: 2.67**  
- Decision Tree → very fast but less accurate  
- SVM, KNN, AdaBoost → weaker performance on noisy real datasets  

Visualizations (scatter, histogram, residuals) shown in Fig. 4.  
:contentReference[oaicite:8]{index=8}

---

## 🧪 6. Backend System (FastAPI)

A FastAPI backend was implemented (page 9) to:

- Receive sensor data  
- Load correct ML model (.joblib)  
- Output **SoC** + **SoH**  
- Provide mode selection (with/without temperature)  
- Integrate easily with frontend visualization  

Fig. 5 shows the GUI for selecting battery type and viewing predictions.  
:contentReference[oaicite:9]{index=9}

---

## 🧠 7. Key Findings

- **Temperature is a critical feature** → improves SoC estimation accuracy  
- Bad cells exhibit higher MAE due to nonlinear degradation  
- Random Forest consistently outperformed all models  
- Hybrid SoH estimation gives stable & realistic degradation trends  
- RUL prediction built on Kaggle dataset generalized well to tester data  
:contentReference[oaicite:10]{index=10}

---

## 🚀 8. Conclusion & Future Work

The research demonstrates a complete end-to-end system:

- Real hardware  
- Real datasets  
- ML regression  
- SoC, SoH, RUL estimation  
- Backend deployment

Future work includes:

- IoT-enabled cloud analytics  
- Real-time remote monitoring  
- LSTM / deep learning for time-series battery degradation  
- Expanding dataset to more chemistries and thermal ranges  
:contentReference[oaicite:11]{index=11}

---

## 🗂️ Files in This Folder

- `First research.pdf` — full research paper with equations, diagrams, tables, and benchmarking results  
- `README.md` — this summary file  
