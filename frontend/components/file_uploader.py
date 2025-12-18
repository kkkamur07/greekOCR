import streamlit as st
from PIL import Image


def image_upload_component():
    st.sidebar.header("📁 Upload Image")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        help="Upload a manuscript image for segmentation"
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            # Display image info
            st.sidebar.success(f"✅ Image loaded successfully")
            st.sidebar.info(f"Size: {image.size[0]} x {image.size[1]} px")
            st.sidebar.info(f"Mode: {image.mode}")
            
            return image
        
        except Exception as e:
            st.sidebar.error(f"❌ Error loading image: {str(e)}")
            return None
    
    return None