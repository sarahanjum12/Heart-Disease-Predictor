from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sklearn.preprocessing import StandardScaler
import joblib
import uvicorn
import pymongo
import numpy as np

# Load the trained SVM model
model = joblib.load('heart_disease_model_svm.joblib')
scaler = joblib.load('scaler.joblib')

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["heart_disease"]
collection = db["predictions"]

print("Connected to MongoDB")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, age: int = Form(...), sex: int = Form(...),
                  cp: int = Form(...), trestbps: int = Form(...),
                  chol: int = Form(...), fbs: int = Form(...),
                  restecg: int = Form(...), thalach: int = Form(...),
                  exang: int = Form(...), oldpeak: float = Form(...),
                  slope: int = Form(...), ca: int = Form(...), thal: int = Form(...)):
    try:
        # Convert NumPy data types to Python native data types
        age = int(age)
        sex = int(sex)
        cp = int(cp)
        trestbps = int(trestbps)
        chol = int(chol)
        fbs = int(fbs)
        restecg = int(restecg)
        thalach = int(thalach)
        exang = int(exang)
        slope = int(slope)
        ca = int(ca)
        thal = int(thal)
        print([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
        
        # Scale the input features
        input_data = scaler.transform(np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]))
        print("Shape: ", input_data.shape)
        print(input_data)
        
        # Perform prediction
        prediction = model.predict(input_data)[0]
        
        print(prediction)
        
        # Calculate risk percentage
        confidence_scores = model.decision_function(input_data)
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        probabilities = sigmoid(confidence_scores)
        risk_percentage = round(probabilities[0]* 100,2)
        print("Risk percentage: ", risk_percentage)
        
        # Store user input, prediction, and risk percentage in MongoDB
        user_input = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
            "prediction": int(prediction),
            "risk_percentage": risk_percentage
        }
        collection.insert_one(user_input)
        print("User input, prediction, and risk percentage stored in the database.")  
        
        if prediction == 1:
            prediction_result = "Heart Disease Risk Predicted"
        else:
            prediction_result = "No Heart Disease Risk Predicted"

        # Render prediction result
        return templates.TemplateResponse("result.html", {"request": request, "prediction_result": prediction_result, "risk_percentage": risk_percentage})
        
    except Exception as e:
        print("Prediction Error:", e)
        return "Internal Server Error: Prediction failed"

# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8888)