from PIL import Image, ImageDraw
import numpy as np
from kraken.binarization import nlbin


def plot_segmentation_pil(
    
    image, lines, regions, 
    show_baseline=True, 
    show_boundary=True, 
    show_bbox=True,
    show_regions=True,
    binarize=True,
    
):
    
    if binarize:
        img_draw = nlbin(image)
        img_draw = img_draw.convert("RGB")
    else : 
        img_draw = image.copy().convert('RGB')
        
    draw = ImageDraw.Draw(img_draw)
    
    # Draw regions
    if show_regions:
        for region in regions:
            if hasattr(region, 'boundary') and region.boundary:
                coords = [coord for point in region.boundary for coord in point]
                draw.polygon(coords, outline=(255, 0, 255), width=6)  # Magenta
    
    # Draw lines
    for idx, line in enumerate(lines):
        # Draw boundary polygon
        if show_boundary:
            boundary_coords = [coord for point in line.boundary for coord in point]
            draw.polygon(boundary_coords, outline='green', width=2)
        
        # Draw baseline
        if show_baseline and len(line.baseline) > 1:
            baseline_coords = [tuple(point) for point in line.baseline]
            draw.line(baseline_coords, fill='blue', width=4)
        
        # Draw bounding box
        if show_bbox:
            boundary = np.array(line.boundary)
            x_min = int(boundary[:, 0].min())
            y_min = int(boundary[:, 1].min())
            x_max = int(boundary[:, 0].max())
            y_max = int(boundary[:, 1].max())
            
            draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=2)
        
        # Add line number
        draw.text(
            (line.baseline[0][0], line.baseline[0][1] - 10), 
            str(idx), 
            fill='yellow',
            stroke_width=2,
            stroke_fill='black'
        )
    
    return img_draw


## do we really need this ? 
def extract_line_images(image, lines):
    line_images = []
    
    for idx, line in enumerate(lines):
        boundary = np.array(line.boundary)
        x_min = int(boundary[:, 0].min())
        y_min = int(boundary[:, 1].min())
        x_max = int(boundary[:, 0].max())
        y_max = int(boundary[:, 1].max())
        
        # Crop line image
        line_img = image.crop((x_min, y_min, x_max, y_max))
        
        line_images.append({
            'id': idx,
            'image': line_img,
            'bbox': (x_min, y_min, x_max, y_max)
        })
    
    return line_images