from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load the model
model = joblib.load('optimized_random_forest_model.joblib')

# Define a Pydantic model that reflects the model's expected features
class PlayerData(BaseModel):
    Ht: float
    forty_yd: float
    Vertical: float
    Bench: float
    Age: float

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
async def root():
    return {"message": "Hello World"}

# Endpoint for making predictions
@app.post("/predict")
async def predict(data: PlayerData):
    try:
        input_data = [data.Ht, data.forty_yd, data.Vertical, data.Bench, data.Age]
        # Assume preprocessing and prediction are synchronous. If you have async operations, use await.
        prediction = model.predict([input_data])
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}
