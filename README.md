# Stock Price Movement Predictor

Predicts **tomorrow's stock movement (UP/DOWN)** using **XGBoost** and **live stock data**.  
Includes a **Streamlit UI**, feature engineering, and a saved model (`stock_predictor.pkl`).

---

## Features
- Live stock data via Yahoo Finance  
- Technical indicators (SMA, EMA, MACD, Volatility, etc.)  
- Prediction with confidence score  
- Interactive Streamlit interface  
- Price chart visualization  

---

## Project Structure

Stock_Predictor/
├── src/
│ ├── app.py
│ ├── predict.py
│ ├── preprocessing.py
│ ├── model_training.py
├── stock_predictor.pkl
├── requirements.txt
└── README.md


---

## Installation
```bash
git clone https://github.com/yourusername/Stock_Predictor.git
cd Stock_Predictor
pip install -r requirements.txt

```

---

## Tech Stack
- Python
- XGBoost
- Streamlit
- Pandas / NumPy
- yfinance
