+import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st
credit_card_data=pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
legit=credit_card_data[credit_card_data.Class==0]
fraud=credit_card_data[credit_card_data.Class==1]
legit_sample=legit.sample(n=492)
new_dataset = pd.concat([legit_sample,fraud],axis=0)
X=new_dataset.drop(columns='Class',axis=1)
Y=new_dataset['Class']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
model=LogisticRegression()
model.fit(X_train,Y_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)


#web app
st.title("Credit Card Fraud Detection Model")
input_df =st.text_input('Enter All Required Features Values')
input_df_splited = input_df.split(',')

submit =st.button("Submit")

if submit:
   features=np.asarray(input_df_splited,dtype=np.float64)
   prediction= model.predict(features.reshape(1,-1))
   
   if prediction[0]==0:
      st.write("Legitimate Transaction")
      
   else:
       st.write("Fraud Transaction")
   
   
