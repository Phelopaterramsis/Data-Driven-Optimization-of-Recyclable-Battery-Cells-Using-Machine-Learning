from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd
import os

class InputData(BaseModel):
    voltage: float
    current: float
    resistance: float
    temp: float | None = None
    capacity: float
    r0: float
    model_file: str
    use_temp: bool

app = FastAPI()

# Mount static files (for CSS and JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/predict")
def predict(data: InputData):
    # Validate model file
    valid_models = ["full2empty", "full2low_bad"]
    if data.model_file not in valid_models:
        raise ValueError(f"Invalid model file. Choose from: {valid_models}")
    
    # Compose model path
    suffix = "_with_temp" if data.use_temp else "_without_temp"
    model_path = f"outputs/{data.model_file}{suffix}_best_model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load model
    model = joblib.load(model_path)

    # Create DataFrame for SOC prediction
    df_data = {
        "Voltage (V)": [data.voltage],
        "Current (mA)": [data.current],
        "Resistance (mOhm)": [data.resistance]
    }
    if data.use_temp:
        df_data["Temp (C)"] = [data.temp]
    
    df = pd.DataFrame(df_data)
    
    # Predict SOC
    soc = model.predict(df)[0]
    
    # Compute SOH
    rated_capacity_mah = 2400.0
    soh_capacity = data.capacity / rated_capacity_mah
    soh_resistance = data.r0 / data.resistance
    soh_combined = 0.5 * (soh_capacity + soh_resistance)
    soh = max(0, min(soh_combined, 1))  # Clip to 0-1
    
    return {"soc": soc, "soh": soh}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
