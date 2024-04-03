# Heart-Disease-Predictor
ntroduction
The Heart Disease Predictor System is a web application built to predict the risk of heart disease for users based on various input parameters such as cholesterol level, fasting blood pressure, etc. The system utilizes a Support Vector Classifier (SVC) model for making predictions and achieves an accuracy of 90%. It also provides confidence scores to determine the probability of predicting the risk percentage.

Features
Predicts the risk of heart disease based on user input parameters.
Utilizes an SVC model for prediction.
Connects to a MongoDB database to store user input and prediction results.
Provides confidence scores for risk percentage prediction.
Web interface for user interaction.
Technologies Used
FastAPI: Web framework for building the API endpoints.
Uvicorn: ASGI server for running the FastAPI application.
MongoDB: NoSQL database for storing user input and prediction results.
Python: Programming language used for backend development.
scikit-learn: Library used for building and training the SVC model.
How to Run
1.Clone the repository:
git clone <repository_url>
2.Pip nstall the required dependencies:
3.Start the MongoDB service and ensure it is running.

Run the FastAPI application with Uvicorn or run main.py ( the uvicorn will run on http://localhost:8888).
