import streamlit as st
import numpy as np
import pickle

st.set_page_config(
    page_title="Multipage App",
    page_icon="👋",
)
model = pickle.load(open('model.pkl','rb'))

def ipm_predict(input_data):
    
    id_np_array = np.asarray(input_data)
    id_reshaped = id_np_array.reshape(1,-1)

    prediction = model.predict(id_reshaped)
    print(prediction)

    if(prediction[0]==0):
        return "Jenis IPM: HIGH"
    elif(prediction[0]==1):
        return "Jenis IPM: LOW"
    elif(prediction[0]==2):
        return "Jenis IPM: Normal"
    else:
        return "Jenis IPM: VERY HIGH"
    
def main():
    
    st.title('KLASIFIKASI JENIS IPM WEB APP')
    
    Harapan_Lama_Sekolah = st.text_input('Harapan Lama Sekolah')
    Pengeluaran_Perkapita = st.text_input('Pengeluaran Perkapita')
    Rerata_Lama_Sekolah = st.text_input('Rerata Lama Sekolah')
    Usia_Harapan_Hidup = st.text_input('Usia Harapan Hidup')
    
    prediksi = ''
    
    if st.button('PREDICT'):
        prediksi = ipm_predict([Harapan_Lama_Sekolah, Pengeluaran_Perkapita, Rerata_Lama_Sekolah, Usia_Harapan_Hidup])
        
    st.success(prediksi)
    
if __name__=='__main__':
    main()

# st.title("Main Page")
# st.sidebar.success("Select a page above.")

# if "my_input" not in st.session_state:
#     st.session_state["my_input"] = ""

# my_input = st.text_input("Input a text here", st.session_state["my_input"])
# submit = st.button("Submit")
# if submit:
#     st.session_state["my_input"] = my_input
#     st.write("You have entered: ", my_input)