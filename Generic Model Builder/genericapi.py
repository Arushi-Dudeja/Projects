#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from typing import List
from trainingdata import readandtrain
import pandas as pd
from fastapi.responses import StreamingResponse
from io import StringIO
import os
import uuid
import pickle
import numpy as np

#from pandas.core.frame import Dataframe 



# In[ ]:


app = FastAPI()


# In[ ]:


class ConfigModel(BaseModel):
    trainingdata: str
    dropcolumns: str
    outputcolumn: str
    testsize: float
    randomstate: int
    sampling: str
    modelname: str
    acceptableaccuracy: float
    modelfile: str
    modelxml: str
        


# In[ ]:


class ResultModel(BaseModel):
    accuracy: float
    conf_matrix : List 
    classification_report : str   
    msg : str 
    model_s1 : str 
    status : int    
         


# In[ ]:


@app.post("/training",response_model=ResultModel)
async def upload_files(datafile: UploadFile = File(...), xmlfile: UploadFile = File(...), outputcolumn:str = Form(...), testsize:float = Form(...), randomstate:int = Form(...), modelname:str = Form(...), acceptableaccuracy:float = Form(...), modelfile:str = Form(...)):
    
    try:
        root_dir = os.path.realpath(__file__ + '/..')
        trainingdata = "{}\\{}".format(root_dir,datafile.filename)
        print("training data file ",trainingdata)
       
        with open(trainingdata, "wb") as f:
            f.write(datafile.file.read())
        
        modelxml = "{}\\{}".format(root_dir,xmlfile.filename)
        print("model file ",modelxml)
       
        with open(modelxml, "wb") as f:
            f.write(xmlfile.file.read())
            
            
        res = await train_model(trainingdata,outputcolumn,testsize,randomstate,modelname,acceptableaccuracy, modelxml, modelfile)    
           
        return res
    except Exception as e:
        print(e)
        return {'accuracy': 0,'conf_matrix':np.zeros((2,2)).tolist(),'classification_report': "",'msg': str(e),'model_s1':"" ,'status':400}
    
    
@app.post("/testing")
async def test_model(datafile: UploadFile = File(...),modelname:str = Form(...) ):
    print(modelname)    
    try:
        root_dir = os.path.realpath(__file__ + '/..')
        testingdata = "{}\\{}".format(root_dir,datafile.filename)
        print("testing data file ",testingdata)
       
        with open(testingdata, "wb") as f:
            f.write(datafile.file.read())
       
        model = pickle.load(open(modelname, 'rb'))
        
        res = process_data(testingdata,model)
           
        return res
    except Exception as e:
        print(e)
        return StreamingResponse(content=iter(list([])), status_code=400)
    


#@app.get("/training", response_model=ResultModel)
#@app.get("/training")

async def train_model(trainingdata,outputcolumn,testsize,randomstate,modelname,acceptableaccuracy,modelxml, modelfile):
    
    #UID = str(uuid.uuid1())
    #modelfile = modelname + "-" + UID +".sav"
    
    #print(modelfile)
    #print(modelname)
    
    accuracy, conf_matrix, classi_report, message, model_s1 = readandtrain(trainingdata," ",outputcolumn,testsize,randomstate,"None",modelname,acceptableaccuracy,modelfile,modelxml)
        
    #print(modelname)
    

    if modelname == 'MultipleLinearRegression' or modelname == 'LinearRegression' or modelname =='RandomForestRegressor' or modelname =='GradientBoostingRegressor'or modelname =='KNNRegressor'or modelname =='DecisionTreeRegressor'or modelname =='SupportVectorRegressor'or modelname =='PolynomialRegression':
        return {'accuracy': accuracy,'conf_matrix':conf_matrix,'classification_report': classi_report,'msg': message,'model_s1':model_s1 ,'status':200};
    else :
        return {'accuracy': accuracy,'conf_matrix': conf_matrix.tolist(),'classification_report': classi_report,'msg': message,'model_s1':model_s1,'status':200 };

    
def process_data(testfile,model):
   
    df = pd.read_csv(testfile, sep=',')
   
    
   
    columns_list = list(df.columns)
   
    predictions = []
   
    for i in range(len(df)):
        feature_array=[]
        for col in columns_list:
            try:
                feature_array.append(df[col][i])
            except TypeError as e:
                print(e)
                feature_array.append(0)
        prediction = model.predict([feature_array])
        print("Prediction of Row {} is {}".format(i, prediction[0]))
        predictions.append(prediction[0])
    # predictions = model.predict([temp_df])
   
    print(predictions)
       
    df["Prediction"] = predictions
   
    stream = StringIO()
    df.to_csv(stream, index=False)
   
    
    res = StreamingResponse(iter([stream.getvalue()]),
                        media_type="text/csv")
    res.headers["Content-Disposition"] = "attachment; filename=output.csv"
    return res    

