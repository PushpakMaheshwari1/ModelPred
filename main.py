import os
from fastapi.encoders import jsonable_encoder
import uvicorn
from fastapi import FastAPI,HTTPException,status
import dill
from crops import crop
import numpy as np
app = FastAPI()
with open('recommendation.pkl', 'rb') as f:
    recommendation = dill.load(f)

@app.get('/')
def index():
    return {'message': "checking machine learning model."}

crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

@app.post('/predict',status_code = status.HTTP_201_CREATED)
def get_parameters(data:crop):
    data = data.dict()
    N = data['N']
    P = data['P']
    K = data['K']
    temperature = data['temperature']
    humidity = data['humidity']
    ph = data['ph']
    rainfall = data['rainfall']

    if not (5 <= temperature <= 50):
        raise HTTPException(status_code=400, detail="Temperature must be between 5 and 50")
    if not (5 <= ph <= 8):
        raise HTTPException(status_code=400, detail="pH must be between 5 and 8")
    if not (10 <= N <= 120):
        raise HTTPException(status_code=400, detail="Nitrogen (N) must be between 10 and 120")
    if not (5 <= P <= 120):
        raise HTTPException(status_code=400, detail="Phosphorus (P) must be between 5 and 120")
    if not (5 <= K <= 120):
        raise HTTPException(status_code=400, detail="Potassium (K) must be between 5 and 120")
    if not (100 <= rainfall <= 2500):
        raise HTTPException(status_code=400, detail="Rainfall must be between 100mm and 2500mm")

    # Make the prediction using your recommendation model
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction_probabilities = recommendation.predict_proba(features)
    
    # Get the indices of the top 3 crops
    top_3_indices = prediction_probabilities.argsort()[0][-3:][::-1]
    
    # Map indices to crop names
    top_3_crops = [crop_dict.get(i + 1, "Unknown Crop") for i in top_3_indices]

    print("top_3_crops",top_3_crops)
    prediction = jsonable_encoder(top_3_crops)
    return {
        'prediction': prediction
    }

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
