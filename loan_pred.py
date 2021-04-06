import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

if __name__=='__main__':
    
    df = pd.read_csv('train_loan.csv')

    #data cleaning
    df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(),inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mean(),inplace=True)

    df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
    df['Married'].fillna(df['Married'].mode()[0],inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
    
    print(df.isnull().sum())

    df['Total_income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    #Analysis

    df['LoanAmount'] = np.log(df['LoanAmount'])
    
    df['ApplicantIncomeLog'] = np.log(df['ApplicantIncome'])
    
    df['Loan_Amount_TermLog'] = np.log(df['Loan_Amount_Term'])
    
    #drop columns
    col = ['Loan_ID','Loan_Amount_Term','ApplicantIncome','CoapplicantIncome','Total_income','LoanAmount']
    df.drop(columns= col,axis = 1,inplace =True)


 

    #preprocessing

    col=['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']    
    le = LabelEncoder()
    for i in col:
      df[i] = le.fit_transform(df[i])  
    
    

    #Train Test data
    
    from sklearn.model_selection import train_test_split

    x = df.loc[:,df.columns != 'Loan_Status']
    y = df.loc[:,df.columns =='Loan_Status']

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)
    print(x_train)


    #classifier function
    def classifier(model,x,y):
        model.fit(x,y)
        print(model.score(x_test,y_test)*100)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier

    lr = LogisticRegression()
    classifier(lr,x_train,y_train)

    clf = RandomForestClassifier()
    classifier(clf,x_train,y_train)
