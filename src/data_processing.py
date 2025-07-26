import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
import os

logger = get_logger(os.path.basename(__file__))

class DataProcessing:
    def __init__(self, input_path,output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.features = None
        
        os.makedirs(self.output_path,exist_ok=True)
        
        logger.info("Data processing Initalized...")
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            logger.info("Data Loaded sucessfully...")
            
        except Exception as e:
            logger.error(f"error while loading the data {e}")
            raise CustomException("failed to load the data",e)
        
    def preprocess(self):
        try:
            self.df["Timestamp"] = pd.to_datetime(self.df["Timestamp"] , errors='coerce')
            categorical_cols = ['Operation_Mode','Efficiency_Status']
            for col in categorical_cols:
                self.df[col] = self.df[col].astype('category')
                
            self.df["Year"] = self.df["Timestamp"].dt.year
            self.df["Month"] = self.df["Timestamp"].dt.month
            self.df["Day"] = self.df["Timestamp"].dt.day

            self.df["Hour"] = self.df["Timestamp"].dt.hour
            
            self.df.drop(columns=["Timestamp","Machine_ID"] , inplace=True)
            
            columns_to_encode = ["Efficiency_Status","Operation_Mode"]
            
            
            
            for column in columns_to_encode:
                le = LabelEncoder()
                self.df[column] = le.fit_transform(self.df[column])
                
            
            logger.info("all basic data preporeccing done...")
                
        except Exception as e:
            logger.error(f"Failed to preprocess data {e}")
            raise CustomException("Failed to preprocess data",e)
        
    
    def split_and_scale_save(self):
        try:
            self.features = [
                        'Operation_Mode', 'Temperature_C', 'Vibration_Hz',
                        'Power_Consumption_kW', 'Network_Latency_ms', 'Packet_Loss_%',
                        'Quality_Control_Defect_Rate_%', 'Production_Speed_units_per_hr',
                        'Predictive_Maintenance_Score', 'Error_Rate_%','Year', 'Month', 'Day', 'Hour'
                    ]
            
            X = self.df[self.features]
            
            y = self.df['Efficiency_Status']  
            
            X_train , X_test , y_train , y_test = train_test_split(X,y, test_size=0.2 , random_state=42 , stratify=y)
            
            scaler = StandardScaler()
          #  print(pd.DataFrame(X_train).info())
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            
            joblib.dump(X_train,os.path.join(self.output_path,'X_train.pkl'))
            joblib.dump(X_test,os.path.join(self.output_path,'X_test.pkl'))
            joblib.dump(y_train,os.path.join(self.output_path,'y_train.pkl'))
            joblib.dump(y_test,os.path.join(self.output_path,'y_test.pkl'))
            
            joblib.dump(scaler,os.path.join(self.output_path,'scaler.pkl'))
            logger.info("data processing done as saved")
            
        except Exception as e:
            logger.error(f"error while data processing....{e}")
            raise CustomException("Failed while data processing....",e)  
        
    def run(self):
        self.load_data()
        self.preprocess()
        self.split_and_scale_save()
            
            
      
if __name__=="__main__":
    input_path = "artifacts/raw/data.csv"
    output_path = "artifacts/processed"
    preocessor = DataProcessing(input_path,output_path)
    preocessor.run()      
            
             
            
            
                                  
        
         
        