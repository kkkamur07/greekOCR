import streamlit as st
from PIL import Image
import io

from components import image_upload_component, visualization_controls, device_selector
from functions import segment_image, extract_data, plot_segmentation_pil, extract_line_images


st.set_page_config(
    page_title="Greek OCR Segmentation",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    
    # Title and description
    st.title("📜 Greek Manuscript Segmentation")
    st.markdown(""" Segment handwritten Greek manuscript images into text lines and regions""")
    
    # Sidebar components
    image = image_upload_component()
    device = device_selector()
    vis_settings = visualization_controls()
    
    # Main content
    if image is None:
        st.info("👈 Please upload an image from the sidebar to begin")
        
        return
    
    # Display original image
    st.subheader("📷 Original Image")
    st.image(image, use_container_width=True)
    
    # Segmentation button
    if st.button("🚀 Run Segmentation", type="primary", use_container_width=True):
        
        with st.spinner("Processing image... This may take a moment."):
            try:
                # Segment image
                segmented = segment_image(image, device=device)
                
                # Extract data
                data = extract_data(segmented)
                
                # Store in session state
                st.session_state['segmented'] = segmented
                st.session_state['data'] = data
                
                st.success(f"✅ Segmentation complete! Found {data['num_lines']} lines and {data['num_regions']} regions")
                
            except Exception as e:
                st.error(f"❌ Error during segmentation: {str(e)}")
                st.stop()
    
    # Display results if available
    if 'segmented' in st.session_state:
        segmented = st.session_state['segmented']
        data = st.session_state['data']
        
        # Statistics
        st.subheader("📊 Segmentation Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Text Lines", data['num_lines'])
        with col2:
            st.metric("Regions", data['num_regions'])
        with col3:
            st.metric("Image Width", image.size[0])
        with col4:
            st.metric("Image Height", image.size[1])
        
        # Visualization
        st.subheader("🎨 Segmentation Visualization")
        
        result_image = plot_segmentation_pil(
            image, 
            segmented.lines, 
            segmented.regions,
            show_baseline=vis_settings['show_baseline'],
            show_boundary=vis_settings['show_boundary'],
            show_bbox=vis_settings['show_bbox'],
            show_regions=vis_settings['show_regions'],
            binarize=vis_settings['binarize']
        )
        
        st.image(result_image, use_container_width=False, width="content")
        
        # Download button for visualization
        buf = io.BytesIO()
        result_image.save(buf, format='PNG')
        st.download_button(
            label="📥 Download Visualization",
            data=buf.getvalue(),
            file_name="segmentation_result.png",
            mime="image/png"
        )
        
        # Extract and display individual lines
        st.subheader("📝 Extracted Text Lines")
        
        line_images = extract_line_images(image, segmented.lines)
        
        # Display in grid
        cols_per_row = 3
        for i in range(0, len(line_images), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(line_images):
                    line_data = line_images[idx]
                    with col:
                        st.image(line_data['image'], caption=f"Line {line_data['id']}", use_container_width=True)
                        st.caption(f"BBox: {line_data['bbox']}")
        
        # Line details (expandable)
        with st.expander("📋 Detailed Line Information"):
            for line_info in data['lines']:
                st.markdown(f"**Line {line_info['id']}**")
                st.json({
                    'bbox': line_info['bbox'],
                    'baseline_points': len(line_info['baseline']),
                    'boundary_points': len(line_info['boundary']),
                    'tags': line_info['tags']
                })
                st.divider()


if __name__ == "__main__":
    main()