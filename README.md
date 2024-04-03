# Heart-Disease-Predictor
#Introduction
The Heart Disease Predictor System is a web application built to predict the risk of heart disease for users based on various input parameters such as cholesterol level, fasting blood pressure, etc. The system utilizes a Support Vector Classifier (SVC) model for making predictions and achieves an accuracy of 90%. It also provides confidence scores to determine the probability of predicting the risk percentage.

#Features
1.Predicts the risk of heart disease based on user input parameters.
2.Utilizes an SVC model for prediction.
3.Connects to a MongoDB database to store user input and prediction results.
4.Provides confidence scores for risk percentage prediction.
5.Web interface for user interaction.

#Technologies Used
1.FastAPI: Web framework for building the API endpoints.
2.Uvicorn: ASGI server for running the FastAPI application.
3.MongoDB: NoSQL database for storing user input and prediction results.
4.Python: Programming language used for backend development.
5.scikit-learn: Library used for building and training the SVC model.

#How to Run

1.Clone the repository:
git clone <repository_url>
2.Pip nstall the required dependencies:
3.Start the MongoDB service and ensure it is running.

Run the FastAPI application with Uvicorn or run main.py ( the uvicorn will run on http://localhost:8888).

Usage

Enter the required parameters for predicting heart disease risk on the web interface.
Click on the "Submit" button.
The system will store the user input in the MongoDB database, make predictions using the SVC model, and return the results along with the risk percentage on the screen.
Predictions are also stored in the MongoDB database.

Contributors

Sarah Anjum

License

This project is licensed under the MIT License.
