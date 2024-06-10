import streamlit as st
import pandas as pd
import pickle


lr_model=pickle.load(open('lr_model.pkl','rb'))
svc_model=pickle.load(open('svc_model.pkl','rb'))

def main():
    st.title("Iris Flower Classification")
    st.write("""
    This app will predicts the **Iris Flower** species based on the input features provided
    """)
    html_temp= """
    <div style="background-color:teal;padding:10px>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['LogisticRegression','SVM']
    options=st.sidebar.selectbox('Select an activity',activities)
    st.subheader(options)
    st.spinner("Hello")
    sl=st.slider("select sepal length",0.0,10.0)
    sw=st.slider("select sepal width",0.0,10.0)
    pl=st.slider("select petal length",0.0,10.0)
    pw=st.slider("select petal width",0.0,10.0)
    inputs=[[sl,sw,pl,pw]]
    if st.button("Classify"):
        if options == 'LinearRegression':
            st.success(lr_model.predict(inputs))
        else:
            st.success(svc_model.predict(inputs))

if __name__=='__main__':
    main()

