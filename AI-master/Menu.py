import streamlit as st 
from PIL import Image

def main():
    
    st.markdown("<h1 style='text-align: center; color: white;'>DETECCION DE EMOCIONES</h1>", unsafe_allow_html=True)
    st.image("imagen.jpg")
    st.sidebar.success("Select a page above. ")
    
   
if __name__ == '__main__':
    main()