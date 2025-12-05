# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags
from itertools import combinations
from google.cloud import vision
from google.api_core.client_options import ClientOptions
import io
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import os
from dotenv import load_dotenv
load_dotenv()



# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="3D Reconstruction Image Analyzer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS for Better UI
# ----------------------------
st.markdown("""
<style>
    .metric-excellent { color: #00cc00; font-weight: bold; }
    .metric-good { color: #88cc00; font-weight: bold; }
    .metric-acceptable { color: #ff9900; font-weight: bold; }
    .metric-poor { color: #ff3333; font-weight: bold; }
    .section-header { 
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 10px;
        border-radius: 5px;
        color: white;
        font-weight: bold;
        margin: 10px 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #667eea;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Calibrated Thresholds
# ----------------------------
LAP_EXCELLENT = 1000
LAP_GOOD = 500
LAP_ACCEPTABLE = 300
TEN_EXCELLENT = 15000
TEN_GOOD = 8000
TEN_ACCEPTABLE = 5000

BRIGHTNESS_LOW = 50
BRIGHTNESS_HIGH = 200
CONTRAST_LOW = 20

MIN_MATCHES_GOOD = 50
MIN_MATCHES_ACCEPTABLE = 30
MIN_OVERLAP_PCT = 20

FEATURE_DENSITY_GOOD = 100
FEATURE_DENSITY_ACCEPTABLE = 50

MIN_RESOLUTION = 800

WEIGHTS = {
    "sharpness": 0.25,
    "brightness_contrast": 0.15,
    "overlap": 0.25,
    "resolution": 0.10,
    "feature_density": 0.15,
    "angle_coverage": 0.10,
}

VIEWS = ["Front", "Back", "Left Side", "Right Side", "Roof/Elevated"]

# Vision API Client
VISION_API_KEY = os.getenv("VISION_API_KEY")  # <-- Read from environment
vision_client = vision.ImageAnnotatorClient(client_options=ClientOptions(api_key=VISION_API_KEY))

# ----------------------------
# Helper Functions
# ----------------------------

def get_quality_color_class(score):
    """Return CSS class based on quality score"""
    if score >= 0.8:
        return "metric-excellent"
    elif score >= 0.6:
        return "metric-good"
    elif score >= 0.4:
        return "metric-acceptable"
    else:
        return "metric-poor"

def get_quality_emoji(score):
    """Return emoji based on quality score"""
    if score >= 0.8:
        return "✅"
    elif score >= 0.6:
        return "✔️"
    elif score >= 0.4:
        return "⚠️"
    else:
        return "❌"

def detect_viewpoint(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes = img_bytes.getvalue()

    vision_image = vision.Image(content=img_bytes)
    response = vision_client.label_detection(image=vision_image)

    labels = [label.description.lower() for label in response.label_annotations]

    if any(l in labels for l in ['front', 'facade', 'door']):
        return 'Front'
    elif any(l in labels for l in ['back', 'garden', 'yard']):
        return 'Back'
    elif any(l in labels for l in ['left', 'side']):
        return 'Left Side'
    elif any(l in labels for l in ['right', 'side']):
        return 'Right Side'
    elif any(l in labels for l in ['roof', 'ceiling', 'top']):
        return 'Roof/Elevated'
    else:
        return 'Unknown'

def correct_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())
        if exif.get(orientation) == 3:
            image = image.rotate(180, expand=True)
        elif exif.get(orientation) == 6:
            image = image.rotate(270, expand=True)
        elif exif.get(orientation) == 8:
            image = image.rotate(90, expand=True)
    except Exception:
        pass
    return image

def convert_to_jpeg(file):
    try:
        img = Image.open(file)
        if img.format in ['CR2', 'HEIC']:
            img = img.convert('RGB')
        img = correct_orientation(img)
        return img
    except Exception as e:
        st.error(f"Failed to open {file.name}: {str(e)}")
        return None

def analyze_sharpness(image):
    gray = np.array(image.convert('L'))
    lap_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    ten_score = np.mean(gx**2 + gy**2)
    
    if lap_score >= LAP_EXCELLENT and ten_score >= TEN_EXCELLENT:
        quality = "Excellent"
        score = 1.0
        ok = True
    elif lap_score >= LAP_GOOD and ten_score >= TEN_GOOD:
        quality = "Good"
        score = 0.8
        ok = True
    elif lap_score >= LAP_ACCEPTABLE and ten_score >= TEN_ACCEPTABLE:
        quality = "Acceptable"
        score = 0.6
        ok = True
    else:
        quality = "Poor (likely blurry)"
        score = 0.3
        ok = False
    
    return {
        'score': score,
        'ok': ok,
        'quality': quality,
        'lap_value': lap_score,
        'ten_value': ten_score
    }

def analyze_brightness_contrast(image):
    gray = np.array(image.convert("L"))
    brightness = gray.mean()
    contrast = gray.std()
    
    issues = []
    if brightness < BRIGHTNESS_LOW:
        issues.append("Too dark")
    elif brightness > BRIGHTNESS_HIGH:
        issues.append("Too bright")
    
    if contrast < CONTRAST_LOW:
        issues.append("Low contrast")
    
    ok = len(issues) == 0
    
    return {
        'brightness': brightness,
        'contrast': contrast,
        'ok': ok,
        'issues': issues
    }

def analyze_feature_density(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints = sift.detect(gray, None)
    num_pixels = gray.shape[0] * gray.shape[1]
    density = (len(keypoints) / num_pixels) * 100000
    
    if density >= FEATURE_DENSITY_GOOD:
        quality = "Good"
        score = 1.0
        ok = True
    elif density >= FEATURE_DENSITY_ACCEPTABLE:
        quality = "Acceptable"
        score = 0.7
        ok = True
    else:
        quality = "Poor (insufficient texture)"
        score = 0.3
        ok = False
    
    return {
        'score': score,
        'ok': ok,
        'quality': quality,
        'num_features': len(keypoints),
        'density': density
    }

def generate_texture_heatmap(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    heatmap = np.uint8(255 * (mag / (mag.max() + 1e-5)))
    texture_score = float(np.mean(heatmap)) / 255.0
    return heatmap, texture_score

def detect_blank_areas(image, threshold=10, min_region_pct=0.2):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    blank_mask = mag < threshold
    blank_pct = np.mean(blank_mask)
    return blank_pct, (blank_pct >= min_region_pct), blank_mask

def detect_glossy_regions(image, sat_threshold=240, min_gloss_pct=0.03):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    glossy_mask = gray > sat_threshold
    glossy_pct = np.mean(glossy_mask)
    return glossy_pct, (glossy_pct >= min_gloss_pct), glossy_mask

def analyze_texture_reconstructibility(sharpness, texture_score, feature_density,
                                       blank_pct, gloss_pct, brightness_ok):
    weights = {
        "sharpness": 0.25,
        "texture": 0.20,
        "features": 0.25,
        "blank_penalty": 0.15,
        "gloss_penalty": 0.10,
        "lighting": 0.05,
    }
    blank_penalty = max(0, 1 - blank_pct*4)
    gloss_penalty = max(0, 1 - gloss_pct*5)
    lighting_score = 1.0 if brightness_ok else 0.5
    
    total = (
        weights["sharpness"] * sharpness +
        weights["texture"] * texture_score +
        weights["features"] * feature_density +
        weights["blank_penalty"] * blank_penalty +
        weights["gloss_penalty"] * gloss_penalty +
        weights["lighting"] * lighting_score
    )
    return float(total)


def estimate_overlap_between_images(img1, img2, min_inliers=50):
    """
    Estimate overlap between two images using SIFT + RANSAC and spatial distribution.
    
    Returns:
        {
            'overlap_pct': float [0-100],
            'num_matches': int,
            'is_sufficient': bool,
            'quality': str,
            'reason': str
        }
    """
    gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return {
            'overlap_pct': 0,
            'num_matches': 0,
            'is_sufficient': False,
            'quality': 'Insufficient features',
            'reason': 'Not enough keypoints to match'
        }

    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    if len(good_matches) < 10:
        return {
            'overlap_pct': 0,
            'num_matches': len(good_matches),
            'is_sufficient': False,
            'quality': f'Only {len(good_matches)} good matches',
            'reason': 'Too few matches for reliable overlap'
        }

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # RANSAC homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inliers = mask.sum() if mask is not None else 0

    # --- Spatial distribution / convex hull ---
    def hull_area(points):
        if len(points) < 3:
            return 1.0  # avoid division by zero
        hull = cv2.convexHull(points)
        return cv2.contourArea(hull)

    src_hull_area = hull_area(src_pts[mask.ravel() == 1])
    dst_hull_area = hull_area(dst_pts[mask.ravel() == 1])
    spatial_overlap_ratio = min(src_hull_area, dst_hull_area) / max(src_hull_area, dst_hull_area)
    spatial_overlap_ratio = np.clip(spatial_overlap_ratio, 0, 1)

    # Combine RANSAC inliers and spatial distribution
    inlier_ratio = min(inliers / 200, 1.0)  # normalize to 0-1, adjust 200 as reference
    overlap_score = 0.7 * spatial_overlap_ratio + 0.3 * inlier_ratio
    overlap_pct = overlap_score * 100

    # Determine quality
    if inliers >= 100 and overlap_pct >= 50:
        quality = "Good overlap"
    elif inliers >= 50 and overlap_pct >= 30:
        quality = "Acceptable overlap"
    else:
        quality = "Insufficient overlap"

    is_sufficient = inliers >= min_inliers and overlap_pct >= 30

    # Construct reason string carefully
    if inliers == 0:
        reason = f"0 geometrically verified matches; images do not overlap"
    else:
        reason = (
            f"{inliers} geometrically verified matches, "
            f"spatial overlap {spatial_overlap_ratio*100:.1f}% "
            f"(~{overlap_pct:.1f}% combined)"
        )

    return {
        'overlap_pct': overlap_pct,
        'num_matches': inliers,
        'is_sufficient': is_sufficient,
        'quality': quality,
        'reason': reason
    }


# ----------------------------
# Streamlit UI
# ----------------------------

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    show_calibration = st.checkbox("Show raw metric values", value=True)
    show_visualizations = st.checkbox("Show texture visualizations", value=True)
    
    st.markdown("---")
    st.header("📚 About")
    st.markdown("""
    This tool evaluates property images for **3D reconstruction** using structure-from-motion (SfM) photogrammetry.
    
    **Key Requirements:**
    - Sharp, well-lit images
    - 60-80% overlap between views
    - Sufficient texture/features
    - Multiple viewing angles
    """)

# Main content
st.title("🏠 Property Image Pre-Screening for 3D Reconstruction")

# Introduction section
with st.expander("ℹ️ Technical Background (Click to expand)", expanded=False):
    st.markdown("""
    ### How Structure-from-Motion (SfM) Works
    
    **Feature Matching:** SfM algorithms identify common points between images to triangulate 3D positions.
    
    **Overlap Requirements:** Adjacent views need 60-80% overlap for reliable depth estimation.
    
    **Feature Density:** Sufficient texture is required for feature point detection and matching. Blank walls fail.
    
    **Sharpness:** Blur degrades feature matching accuracy and reconstruction quality.
    
    **Lighting:** Consistent lighting helps feature detection; extreme brightness/darkness is problematic.
    
    **Viewing Angles:** Complete 360° coverage with 20-30° increments provides best results.
    
    ⚠️ *Note: Automatic viewing angle estimation requires camera calibration. Manual verification recommended.*
    """)

# File upload
uploaded_files = st.file_uploader(
    "📤 Upload property images (JPG, PNG, HEIC, CR2)",
    type=['jpg', 'jpeg', 'png', 'heic', 'cr2'],
    accept_multiple_files=True,
    help="Upload 8-12 images from different angles for best 3D reconstruction results"
)

if uploaded_files:
    if len(uploaded_files) == 1:
        st.markdown('<div class="warning-box">⚠️ <b>Single image uploaded.</b> 3D reconstruction requires multiple overlapping views from different angles. Upload 8-12 images for best results.</div>', unsafe_allow_html=True)
    
    # Load images
    images = []
    filenames = []
    for f in uploaded_files:
        img = convert_to_jpeg(f)
        if img is not None:
            images.append(img)
            filenames.append(f.name)
    
    if not images:
        st.error("No valid images loaded.")
        st.stop()
    
    # Progress indicator
    st.markdown('<div class="section-header">📊 Analyzing Images...</div>', unsafe_allow_html=True)
    analysis_progress = st.progress(0)
    
    # Analyze each image
    results = []
    
    for idx, img in enumerate(images):
        # Update progress
        analysis_progress.progress((idx + 1) / len(images))
        
        # Analysis
        width, height = img.size
        min_side = min(width, height)
        
        if min_side < 500:
            resolution_ok = False
            resolution_score = 0.2
        elif min_side < 800:
            resolution_ok = True
            resolution_score = 0.5
        else:
            resolution_ok = True
            resolution_score = 1.0
        
        sharpness_result = analyze_sharpness(img)
        lighting_result = analyze_brightness_contrast(img)
        feature_result = analyze_feature_density(img)
        heatmap, texture_score = generate_texture_heatmap(img)
        blank_pct, blank_flag, blank_mask = detect_blank_areas(img)
        gloss_pct, gloss_flag, gloss_mask = detect_glossy_regions(img)
        
        reconstructibility = analyze_texture_reconstructibility(
            sharpness_result['score'],
            texture_score,
            feature_result['score'],
            blank_pct,
            gloss_pct,
            lighting_result['ok']
        )
        
        view_label = detect_viewpoint(img)
        
        result = {
            'filename': filenames[idx],
            'image': img,
            'view': view_label,
            'width': width,
            'height': height,
            'resolution_ok': resolution_ok,
            'resolution_score': resolution_score,
            'sharpness': sharpness_result,
            'lighting': lighting_result,
            'features': feature_result,
            'texture_score': texture_score,
            'blank_pct': blank_pct,
            'blank_flag': blank_flag,
            'gloss_pct': gloss_pct,
            'gloss_flag': gloss_flag,
            'reconstructibility': reconstructibility,
            'heatmap': heatmap,
            'blank_mask': blank_mask,
            'gloss_mask': gloss_mask,
            'best_overlap': 0,
        }
        results.append(result)
    
    analysis_progress.empty()
    
    # Individual Image Analysis
    st.markdown('<div class="section-header">🔍 Individual Image Analysis</div>', unsafe_allow_html=True)
    
    for idx, r in enumerate(results):
        with st.expander(f"📷 {r['filename']} - {get_quality_emoji(r['sharpness']['score'])} {r['sharpness']['quality']}", expanded=(idx == 0)):
            col1, col2 = st.columns([1, 2])
            
            # -----------------------------
            # LEFT COLUMN (Image + View + Reconstructibility)
            # -----------------------------
            with col1:
                st.image(r['image'], use_container_width=True)

                # More prominent Detected View
                st.markdown(
                    f"""
                    <div style="font-size:1.1rem; font-weight:600; margin-top:0.5rem;">
                        Detected View: <span style="font-weight:700;">{r['view']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Reconstructibility Score directly under image
                recon_class = get_quality_color_class(r['reconstructibility'])
                st.markdown(
                    f"""
                    <div style="margin-top:0.5rem;">
                        <strong>Reconstructibility Score:</strong><br>
                        <span class="{recon_class}" style="font-size:1.2rem;">
                            {r['reconstructibility']:.2f} / 1.0
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # -----------------------------
            # RIGHT COLUMN (Metrics)
            # -----------------------------
            with col2:
                # Metrics in columns
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                # Sharpness
                with metric_col1:
                    sharp_class = get_quality_color_class(r['sharpness']['score'])
                    st.markdown("**Sharpness**")
                    st.markdown(f"<span class='{sharp_class}'>{r['sharpness']['quality']}</span>", unsafe_allow_html=True)
                    if show_calibration:
                        st.caption(f"Lap: {r['sharpness']['lap_value']:.0f} | Ten: {r['sharpness']['ten_value']:.0f}")
                
                # Lighting
                with metric_col2:
                    light_class = get_quality_color_class(1.0 if r['lighting']['ok'] else 0.3)
                    st.markdown("**Lighting**")
                    st.markdown(f"<span class='{light_class}'>{'Good' if r['lighting']['ok'] else ', '.join(r['lighting']['issues'])}</span>", unsafe_allow_html=True)
                    if show_calibration:
                        st.caption(f"Bright: {r['lighting']['brightness']:.0f} | Contrast: {r['lighting']['contrast']:.0f}")
                
                # Features
                with metric_col3:
                    feat_class = get_quality_color_class(r['features']['score'])
                    st.markdown("**Features**")
                    st.markdown(f"<span class='{feat_class}'>{r['features']['quality']}</span>", unsafe_allow_html=True)
                    st.caption(f"{r['features']['num_features']} keypoints")
                
                # Resolution (only)
                st.markdown("---")
                res_emoji = "✅" if r['resolution_score'] >= 1.0 else "⚠️" if r['resolution_score'] >= 0.5 else "❌"
                st.markdown(f"**Resolution:** {res_emoji} {r['width']}×{r['height']}")

                # Texture Analysis
                st.markdown("---")
                st.markdown("**Texture Analysis**")
                
                tex_col1, tex_col2 = st.columns(2)
                with tex_col1:
                    blank_color = "red" if r['blank_flag'] else "green"
                    st.markdown(f"Blank Areas: <span style='color:{blank_color}'>{r['blank_pct']*100:.1f}%</span>", unsafe_allow_html=True)
                
                with tex_col2:
                    gloss_color = "red" if r['gloss_flag'] else "green"
                    st.markdown(f"Glossy Areas: <span style='color:{gloss_color}'>{r['gloss_pct']*100:.1f}%</span>", unsafe_allow_html=True)
                
                if r['blank_flag']:
                    st.warning("⚠️ Large blank areas detected → low reconstructibility")
                if r['gloss_flag']:
                    st.warning("⚠️ Significant glossy regions → feature detection will fail")
                
                # Visualizations
                if show_visualizations:
                    st.markdown("**Texture Visualizations**")
                    viz_col1, viz_col2, viz_col3 = st.columns(3)
                    with viz_col1:
                        st.image(r['heatmap'], caption="Texture Heatmap", use_container_width=True)
                    with viz_col2:
                        st.image((r['blank_mask'] * 255).astype(np.uint8), caption="Blank Areas", use_container_width=True)
                    with viz_col3:
                        st.image((r['gloss_mask'] * 255).astype(np.uint8), caption="Glossy Areas", use_container_width=True)




    # Overlap Analysis
    if len(images) > 1:
        st.markdown('<div class="section-header">🔗 Image Overlap Analysis</div>', unsafe_allow_html=True)

        with st.spinner("Analyzing feature matching between image pairs..."):
            overlap_matrix = []
            pairs = list(combinations(range(len(images)), 2))
            
            for i, j in pairs:
                overlap_info = estimate_overlap_between_images(images[i], images[j])
                overlap_matrix.append((i, j, overlap_info))
                results[i]['best_overlap'] = max(results[i]['best_overlap'], overlap_info['num_matches'])
                results[j]['best_overlap'] = max(results[j]['best_overlap'], overlap_info['num_matches'])

        # Create overlap visualization DataFrame
        overlap_data = []
        for i, j, info in overlap_matrix:
            # Instead of filenames, store indices for reference
            overlap_data.append({
                'Image1_idx': i,
                'Image2_idx': j,
                'Matches': info['num_matches'],
                'Quality': info['quality'],
                'Sufficient': info['is_sufficient']
            })

        df_overlap = pd.DataFrame(overlap_data)

        # Bar chart (can still use indices as labels)
        fig_overlap = px.bar(
            df_overlap,
            x=[f"{i}-{j}" for i, j in zip(df_overlap['Image1_idx'], df_overlap['Image2_idx'])],
            y='Matches',
            color='Sufficient',
            color_discrete_map={True: 'green', False: 'red'},
            title="Pairwise Feature Matches"
        )
        fig_overlap.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Good threshold (50)")
        fig_overlap.update_layout(showlegend=True, xaxis_tickangle=-45)
        st.plotly_chart(fig_overlap, use_container_width=True)

        # Detailed table with thumbnails
        with st.expander("📋 Detailed Overlap Results"):
            for i, j, info in overlap_matrix:
                color = 'green' if info['is_sufficient'] else 'orange'
                cols = st.columns([1,1,4])
                with cols[0]:
                    st.image(images[i], width=80)
                with cols[1]:
                    st.image(images[j], width=80)
                with cols[2]:
                    st.markdown(f"<span style='color:{color}'>{info['quality']}</span> - {info['reason']}", unsafe_allow_html=True)
    

    # Viewing Angle Coverage
    st.markdown('<div class="section-header">📐 Viewing Angle Coverage</div>', unsafe_allow_html=True)

    # Count images per view
    view_counts = {view: 0 for view in VIEWS}
    for r in results:
        if 'view' in r and r['view'] in view_counts:
            view_counts[r['view']] += 1

    # Compute coverage %
    coverage_pct = (sum(1 for count in view_counts.values() if count > 0) / len(VIEWS)) * 100

    # Display coverage visually
    for view, count in view_counts.items():
        st.text(f"{view}: {count} image(s)")

    st.text(f"Overall coverage: {coverage_pct:.1f}%")


    # Coverage visualization
    fig_coverage = go.Figure([go.Bar(
        x=list(view_counts.keys()),
        y=list(view_counts.values()),
        marker_color=['#00cc00' if c > 0 else '#ff3333' for c in view_counts.values()],
        text=list(view_counts.values()),
        textposition='auto',
    )])
    fig_coverage.update_layout(
        title=f"Viewpoint Coverage: {coverage_pct:.0f}% of recommended viewpoints",
        yaxis_title="Number of Images",
        xaxis_title="Viewpoint",
        yaxis=dict(dtick=1),
        showlegend=False
    )
    st.plotly_chart(fig_coverage, use_container_width=True)
    
    missing = [v for v, c in view_counts.items() if c == 0]
    if missing:
        st.markdown(f'<div class="warning-box">⚠️ <b>Missing viewpoints:</b> {", ".join(missing)}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">✅ <b>All recommended viewpoints covered!</b></div>', unsafe_allow_html=True)
    
    st.info("💡 **Best practice:** Capture images from all sides with 20-30° angle increments for complete 360° coverage.")
    
    # 3D Reconstruction Readiness
    st.markdown('<div class="section-header">🎯 3D Reconstruction Readiness Scores</div>', unsafe_allow_html=True)
    
    readiness_data = []
    
    for r in results:
        score = 0
        score += WEIGHTS["sharpness"] * r['sharpness']['score']
        score += WEIGHTS["brightness_contrast"] * (1.0 if r['lighting']['ok'] else 0.5)
        score += WEIGHTS["resolution"] * r['resolution_score']
        score += WEIGHTS["feature_density"] * r['features']['score']
        score += 0.15 * r['reconstructibility']
        
        if len(images) > 1:
            if r['best_overlap'] >= MIN_MATCHES_GOOD:
                overlap_score = 1.0
            elif r['best_overlap'] >= MIN_MATCHES_ACCEPTABLE:
                overlap_score = 0.7
            else:
                overlap_score = 0.3
        else:
            overlap_score = 0.0
        score += WEIGHTS["overlap"] * overlap_score
        
        coverage_score = len([c for c in view_counts.values() if c > 0]) / len(VIEWS)
        score += WEIGHTS["angle_coverage"] * coverage_score
        
        final_score = min(score * 100, 100)
        r['final_score'] = final_score
        
        if final_score >= 75:
            readiness = "Excellent"
            status = "✅ Ready for 3D reconstruction"
            color = "#00cc00"
        elif final_score >= 60:
            readiness = "Good"
            status = "✔️ Minor improvements recommended"
            color = "#88cc00"
        elif final_score >= 40:
            readiness = "Fair"
            status = "⚠️ Several improvements needed"
            color = "#ff9900"
        else:
            readiness = "Poor"
            status = "❌ Not suitable for 3D reconstruction"
            color = "#ff3333"
        
        readiness_data.append({
            'Filename': r['filename'][:30] + '...' if len(r['filename']) > 30 else r['filename'],
            'Score': final_score,
            'Status': readiness,
            'Color': color
        })
    
    # Create readiness chart
    df_readiness = pd.DataFrame(readiness_data)
    fig_readiness = go.Figure([go.Bar(
        x=df_readiness['Filename'],
        y=df_readiness['Score'],
        marker_color=df_readiness['Color'],
        text=[f"{s:.1f}" for s in df_readiness['Score']],
        textposition='auto',
    )])
    fig_readiness.add_hline(y=75, line_dash="dash", line_color="green", annotation_text="Excellent (75+)")
    fig_readiness.add_hline(y=60, line_dash="dash", line_color="orange", annotation_text="Good (60+)")
    fig_readiness.add_hline(y=40, line_dash="dash", line_color="red", annotation_text="Fair (40+)")
    fig_readiness.update_layout(
        title="Individual Image Readiness Scores",
        yaxis_title="Readiness Score (0-100)",
        xaxis_title="Image",
        yaxis=dict(range=[0, 100]),
        showlegend=False,
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig_readiness, use_container_width=True)
    
    # Detailed scores table
    with st.expander("📊 Detailed Score Breakdown"):
        for r in readiness_data:
            st.markdown(f"**{r['Filename']}**: <span style='color:{r['Color']}'>{r['Score']:.1f}/100 - {r['Status']}</span>", unsafe_allow_html=True)
    
    # Overall Assessment
    st.markdown('<div class="section-header">📈 Overall Dataset Assessment</div>', unsafe_allow_html=True)

    # Compute raw average score from images
    avg_score = np.mean([r['final_score'] for r in results])

    # Count covered views
    num_views_covered = len([c for c in view_counts.values() if c > 0])
    total_views = len(VIEWS)

    # Penalize for missing views: reduce score proportional to missing fraction
    coverage_factor = num_views_covered / total_views
    adjusted_score = avg_score * coverage_factor

    # Summary metrics
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    # Function to determine color based on value
    def color_for_score(value, max_value=100):
        pct = value / max_value
        if pct >= 0.75:
            return "green"
        elif pct >= 0.5:
            return "orange"
        else:
            return "red"

    # Calculate colors
    avg_color = color_for_score(adjusted_score)
    images_color = "green" if len(images) > 0 else "red"
    coverage_pct = num_views_covered / total_views
    coverage_color = "green" if coverage_pct >= 0.8 else "orange" if coverage_pct >= 0.5 else "red"
    excellent_count = sum(1 for r in results if r['final_score'] >= 75)
    excellent_color = "green" if excellent_count == len(images) else "orange" if excellent_count >= len(images)/2 else "red"

    # Metrics in columns
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.markdown(f'<div style="font-size:20px; font-weight:bold; color:{avg_color}">Average Score: {adjusted_score:.1f}/100</div>', unsafe_allow_html=True)

    with metric_col2:
        st.markdown(f'<div style="font-size:20px; font-weight:bold; color:{images_color}">Images Analyzed: {len(images)}</div>', unsafe_allow_html=True)

    with metric_col3:
        st.markdown(f'<div style="font-size:20px; font-weight:bold; color:{coverage_color}">Viewpoints Covered: {num_views_covered}/{total_views}</div>', unsafe_allow_html=True)

    with metric_col4:
        st.markdown(f'<div style="font-size:20px; font-weight:bold; color:{excellent_color}">Excellent Images: {excellent_count}/{len(images)}</div>', unsafe_allow_html=True)



    # Overall verdict
    st.markdown("---")
    if adjusted_score >= 75 and num_views_covered >= 4:
        st.markdown(f'<div class="success-box"><h3>✅ Excellent Dataset</h3><p>Average score: <b>{adjusted_score:.1f}/100</b><br>This image set is well-suited for high-quality 3D reconstruction. All key requirements are met.</p></div>', unsafe_allow_html=True)
    elif adjusted_score >= 60 and num_views_covered >= 3:
        st.markdown(f'<div class="warning-box"><h3>✔️ Good Dataset</h3><p>Average score: <b>{adjusted_score:.1f}/100</b><br>This image set will produce acceptable 3D reconstruction. Minor improvements would enhance quality.</p></div>', unsafe_allow_html=True)
    elif adjusted_score >= 40:
        st.markdown(f'<div class="warning-box"><h3>⚠️ Fair Dataset</h3><p>Average score: <b>{adjusted_score:.1f}/100</b><br>3D reconstruction possible but quality may be limited. Several improvements recommended.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="error-box"><h3>❌ Insufficient Dataset</h3><p>Average score: <b>{adjusted_score:.1f}/100</b><br>This image set is not suitable for reliable 3D reconstruction. Please address the issues below.</p></div>', unsafe_allow_html=True)

    # Recommendations
    st.markdown('<div class="section-header">💡 Actionable Recommendations</div>', unsafe_allow_html=True)
    
    recommendations = []
    
    blurry_count = sum(1 for r in results if not r['sharpness']['ok'])
    if blurry_count > 0:
        recommendations.append({
            'icon': '📸',
            'priority': 'High',
            'issue': f'{blurry_count} blurry image(s)',
            'action': 'Retake using tripod or faster shutter speed. Ensure autofocus is working correctly.'
        })
    
    low_texture = sum(1 for r in results if not r['features']['ok'])
    if low_texture > 0:
        recommendations.append({
            'icon': '🎨',
            'priority': 'High',
            'issue': f'{low_texture} image(s) with insufficient texture',
            'action': 'Blank walls and uniform surfaces lack features for matching. Ensure surfaces have visual detail.'
        })
    
    if len(images) > 1:
        poor_overlap = sum(1 for r in results if r['best_overlap'] < MIN_MATCHES_ACCEPTABLE)
        if poor_overlap > 0:
            recommendations.append({
                'icon': '🔗',
                'priority': 'High',
                'issue': f'{poor_overlap} image(s) with insufficient overlap',
                'action': 'Ensure 60-80% overlap between adjacent views. Take more intermediate shots between existing angles.'
            })
    
    if num_views_covered < 4:
        recommendations.append({
            'icon': '📐',
            'priority': 'Medium',
            'issue': f'Only {num_views_covered}/5 viewpoints covered',
            'action': f'Capture images from: {", ".join(missing)}'
        })
    
    low_res = sum(1 for r in results if r['resolution_score'] < 0.5)
    if low_res > 0:
        recommendations.append({
            'icon': '🖼️',
            'priority': 'Medium',
            'issue': f'{low_res} low resolution image(s)',
            'action': 'Use camera with higher resolution or move closer to subject.'
        })
    
    lighting_issues = sum(1 for r in results if not r['lighting']['ok'])
    if lighting_issues > 0:
        recommendations.append({
            'icon': '💡',
            'priority': 'Low',
            'issue': f'{lighting_issues} image(s) with lighting issues',
            'action': 'Shoot during consistent lighting conditions. Avoid harsh shadows and overexposure.'
        })
    
    if not recommendations:
        st.markdown('<div class="success-box"><h4>✅ No issues detected!</h4><p>Your image set meets all requirements for 3D reconstruction.</p></div>', unsafe_allow_html=True)
    else:
        for rec in recommendations:
            priority_color = {'High': '#ff3333', 'Medium': '#ff9900', 'Low': '#88cc00'}[rec['priority']]
            st.markdown(f"""
            <div class="info-box">
                <h4>{rec['icon']} <span style='color:{priority_color}'>[{rec['priority']} Priority]</span> {rec['issue']}</h4>
                <p><b>Action:</b> {rec['action']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Export option
    st.markdown("---")
    if st.button("📥 Export Analysis Report (CSV)"):
        export_data = []
        for r in results:
            export_data.append({
                'Filename': r['filename'],
                'View': r['view'],
                'Resolution': f"{r['width']}x{r['height']}",
                'Sharpness_Score': r['sharpness']['score'],
                'Feature_Count': r['features']['num_features'],
                'Reconstructibility': r['reconstructibility'],
                'Best_Overlap_Matches': r['best_overlap'],
                'Final_Score': r['final_score']
            })
        
        df_export = pd.DataFrame(export_data)
        csv = df_export.to_csv(index=False)
        st.download_button(
            label="Download CSV Report",
            data=csv,
            file_name="3d_reconstruction_analysis.csv",
            mime="text/csv"
        )

else:
    # Welcome screen
    st.markdown('<div class="info-box"><h3>👋 Welcome!</h3><p>Upload property images to begin analysis for 3D reconstruction readiness.</p></div>', unsafe_allow_html=True)
    
    st.markdown("### 📋 Quick Start Guide")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1️⃣ Capture Images**")
        st.markdown("- Take 8-12 photos")
        st.markdown("- Cover all sides")
        st.markdown("- Ensure 60-80% overlap")
    
    with col2:
        st.markdown("**2️⃣ Upload & Analyze**")
        st.markdown("- Upload all images")
        st.markdown("- Wait for analysis")
        st.markdown("- Review results")
    
    with col3:
        st.markdown("**3️⃣ Take Action**")
        st.markdown("- Follow recommendations")
        st.markdown("- Retake problematic shots")
        st.markdown("- Achieve 75+ score")