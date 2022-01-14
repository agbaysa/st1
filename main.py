# Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st

import warnings
warnings.filterwarnings('ignore')


# Read the data
df = pd.read_csv('https://raw.githubusercontent.com/agbaysa/st1/main/data_tree.csv')

# Features Enginerring
df['job'].loc[df['job'] == 'With Business'] = 0
df['job'].loc[df['job'] == 'Employed'] = 1

df['status'].loc[df['status'] == 'Single'] = 0
df['status'].loc[df['status'] == 'Married'] = 1

# Split
y = df['good_bad']
X = df[['age','job','mo_income','no_family','status']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


# Fit model
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit
st.title('Scoring Loan Applications Using Streamlit')
st.image('1vb_logo.jpg', use_column_width='always')
st.subheader('This web service app scores the loan application and determines if it is good or bad. Please enter the details of the loan applicant below and click the SUBMIT button.')


age = st.number_input(label='Enter Age of applicant:', value=30)
job = st.selectbox('Select Job Type of applicant:', ('With Business','Employed'))
mo_income = st.number_input(label='Enter Monthly Income of applicant (no commas):', value=30000)
status = st.selectbox('Select Job Type of applicant:', ('Single','Married'))
no_family = st.number_input(label='Enter Number of Family Members (including applicant):', value=4)



if st.button('Submit'):
    # Convert values
    if job == 'With Business':
        job = 0
    else:
        job = 1

    if status == 'Single':
        status = 0
    else:
        status = 1


    df_predict = {'age': [age], 'job': [job], 'mo_income': mo_income, 'no_family': no_family, 'status': [status]}
    df_predict = pd.DataFrame(df_predict)

    st.write('The Loan Application Raw Score is:')
    my_score = model.predict_proba(df_predict)[0][1]
    st.write(my_score)

    if my_score < 0.6:
        st.write('The raw score is less than the cut-off score of 0.6')
        st.write('KINDLY REVIEW THE APPLICATION AND SEEK APPROVAL PRIOR TO BOOKING.')
    else:
        st.write('The raw score is greater than the cut-off score of 0.6.')
        st.write('KINDLY BOOK THE APPLICATION.')

