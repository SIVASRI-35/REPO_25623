import streamlit as st
import pandas as pd
#import pickle
from sklearn.ensemble import RandomForestClassifier
import base64

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-attachment: fixed;
    background-position: center center;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


#model = pickle.load(open('model.pkl','rb'))
df = pd.read_csv("crop_dataset3.csv",encoding='cp1252')
data=df.dropna()

data["Averageduration"]=data[['STARTING DURATION','ENDING DURATION']].mean(axis=1)
data=data.drop(['STARTING DURATION','ENDING DURATION'],axis=1)

xseed1=data['X'].unique()
xseed2=data['Y'].unique()

#print("seed1:",xseed1)
#print("seed2:",xseed2)

# Select the features to predict and convert them to string data type
features_to_predict = ['TYPES OF PADDY','PERIOD', 'AVERAGE YIELD(Kg/ha)','1000 GRAIN WEIGHT(g)','GRAIN TYPE','HABIT','RICE COLOR','SPECIAL FEATURES',
'STARTING MONTH', 'ENDING MONTH', 'DISTRICT','RAINFALL ACTUAL(in mm)','RAINFALL NORMAL(in mm)','TMAX (°F)','TMIN (°F)','Averageduration']
data[features_to_predict] = data[features_to_predict].astype(str)


X = data[['X', 'Y']]
y = data[features_to_predict]

X_encoded = pd.get_dummies(X, columns=['X', 'Y'])

# Train the model
model = RandomForestClassifier()
model.fit(X_encoded, y)


def prediction_fun(x1,y1):
  new_sample = pd.DataFrame({'X': [x1],'Y': [y1]})
  new_sample_encoded = pd.get_dummies(new_sample, columns=['X', 'Y'])
  # Reindex the new sample encoded features to match the training data columns
  new_sample_encoded = new_sample_encoded.reindex(columns=X_encoded.columns, fill_value=0)
  predicted_features = model.predict(new_sample_encoded)
  #print("\nPredicted features:\n")
  st.subheader("PREDICTED FEATURES")
  for feature, prediction in zip(features_to_predict, predicted_features[0]):
    st.write(feature,":",prediction)
  #print(feature + ":", prediction)

set_background("img10.png")

st.header("CROP & ITS CHARACTERISTICS PREDICTION:black")

col1,col2=st.columns(2)
with col1:
   st.subheader("_Parent seed1_")#seed1=st.text_input("_Enter seed1:_")
   seed1=st.radio("Select seed1",xseed1)
with col2:
   st.subheader("_Parent seed2_")#seed2=st.text_input("Enter seed2:")
   seed2=st.radio("Select seed2",xseed2)


sub=st.button("SUBMIT")

if(sub):
   st.write("PARENT SEED1:",seed1)
   st.write("PARENT SEED2:",seed2)
   prediction_fun(seed1,seed2) 
