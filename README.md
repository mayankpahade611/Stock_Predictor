# Stock Price Movement Predictor

Predicts **tomorrow's stock movement (UP/DOWN)** using **XGBoost** and **live stock data**.  
Includes a **Streamlit UI**, feature engineering, and a saved model (`stock_predictor.pkl`).
[Click for app demo](https://stockpredictor-pirvb8p7vk7amqghzsatrs.streamlit.app/)

---


## Features
- Live stock data via Yahoo Finance  
- Technical indicators (SMA, EMA, MACD, Volatility, etc.)  
- Prediction with confidence score  
- Interactive Streamlit interface  
- Price chart visualization  



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
