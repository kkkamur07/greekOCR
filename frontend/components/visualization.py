
import streamlit as st
import torch


def visualization_controls():

    st.sidebar.header("🎨 Visualization Options")
    
    settings = {
        'show_baseline': st.sidebar.checkbox("Show Baselines", value=True),
        'show_boundary': st.sidebar.checkbox("Show Boundaries", value=True),
        'show_bbox': st.sidebar.checkbox("Show Bounding Boxes", value=True),
        'show_regions': st.sidebar.checkbox("Show Regions", value=True),
        'binarize': st.sidebar.checkbox("Binarize Image", value=True)
    }
    
    return settings


def device_selector():
    st.sidebar.header("⚙️ Settings")
    has_cuda = torch.cuda.is_available()
    
    if has_cuda:
        device = st.sidebar.radio(
            "Processing Device",
            options=['cpu', 'cuda'],
            index=1,
            help="CUDA is available - using GPU will be faster"
        )
        st.sidebar.success(f"✅ Using: {device.upper()}")
    else:
        device = 'cpu'
        st.sidebar.info("ℹ️ Using: CPU (CUDA not available)")
    
    return device