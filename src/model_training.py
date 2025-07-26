import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.logger import get_logger
from src.custom_exception import CustomException
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


logger = get_logger(os.path.basename(__file__))

class ModelTraining:
    
    def __init__(self,processed_path,model_ouput_path):
        self.processed_path = processed_path
        self.model_ouput_path = model_ouput_path
        self.clf = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        os.makedirs(self.model_ouput_path,exist_ok=True)
        logger.info("model training initialzed ...")
        
    def load_data(self):
        try:
            self.X_train = joblib.load(os.path.join(self.processed_path,"X_train.pkl"))
            self.X_test = joblib.load(os.path.join(self.processed_path,"X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.processed_path,"y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.processed_path,"y_test.pkl"))
            
            logger.info("data loaded sucessfully...")
            
        except Exception as e:
            logger.error("failed to load the data",e)
            raise CustomException("failed to load the model",e)
        
        
    def train_model(self):
        try:
            self.clf = LogisticRegression(random_state=42,max_iter=100)
            self.clf.fit(self.X_train,self.y_train)
            
            joblib.dump(self.clf,os.path.join(self.model_ouput_path,"model.pkl"))
            
            logger.info("model trainined and saved")
        except Exception as e:
            logger.error("failed to train the model",e)
            raise CustomException("failed to train the model",e)
        
    def evaluate_model(self):
        try:
            y_pred = self.clf.predict(self.X_test)
            
            accuracy = accuracy_score(self.y_test,y_pred)
            
            precision = precision_score(self.y_test,y_pred,average = 'weighted')
            recall = recall_score(self.y_test,y_pred,average = 'weighted')
            f1 = f1_score(self.y_test,y_pred,average = 'weighted')
            logger.info(f"accuracy : {accuracy}")
            logger.info(f"accuracy : {precision}")
            logger.info(f"accuracy : {recall}")
            logger.info(f"accuracy : {f1}")
            
            return accuracy,precision,recall,f1
        except Exception as e:
            logger.error("failed evaluate the model ",e)
            raise CustomException("failed evaluate the model",e)
            
    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()
        
if __name__ == '__main__':
    trainer = ModelTraining("artifacts/processed/",'artifacts/models/')
    trainer.run()
    

        
        