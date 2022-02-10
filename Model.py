# 1. Library imports
import pandas as pd 
from pydantic import BaseModel
import joblib
from sklearn.ensemble import RandomForestRegressor

# 2. Class which describes a single flower measurements
class IrisSpecies(BaseModel):
    Sex :float
    Length :float
    Diameter :float
    Height :float
    Wholeweight: float
    Shuckedweight: float
    Visceraweight: float
    Shellweight: float
    # sepal_length: float 
    # sepal_width: float 
    # petal_length: float 
    # petal_width: float


# 3. Class for training the model and making predictions
class IrisModel:
    # 6. Class constructor, loads the dataset and loads the model
    #    if exists. If not, calls the _train_model method and 
    #    saves the model
    def __init__(self):
        # self.df = pd.read_csv('iris.csv')
        self.model_fname_ = 'model_job_rf.joblib'
        self.model = joblib.load(self.model_fname_)
        try:
            with open(self.model_fname_,"rb") as f:
            # self.model= pickle.load(f)
                self.model = joblib.load(self.model_fname_)
                # print(car_obj_2)
        
        except Exception as _:
                self.model = self._train_model()
                joblib.dump(self.model, self.model_fname_)
        

    #4. Perform model training using the RandomForest classifier
    def _train_model(self):
        X = self.df.drop(columns='Rings',axis=1)
        y = self.df['Rings']
        rfc = RandomForestRegressor()
        model = rfc.fit(X, y)
        return model


    # 5. Make a prediction based on the user-entered data
    #    Returns the predicted species with its respective probability
    def predict_species(self,Sex, Length, Diameter, Height, Wholeweight, Shuckedweight, Visceraweight, Shellweight):
        data_in = [[Sex, Length, Diameter, Height, Wholeweight, Shuckedweight, Visceraweight, Shellweight]]
        prediction = self.model.predict(data_in)
        # probability = self.model.predict_proba(data_in).max()
        return prediction[0]