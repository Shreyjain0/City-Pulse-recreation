"""
CityPulse Annotation Tool
Interactive web interface for labeling urban change points in street view time series
"""

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict
import hashlib
import shutil

# Page configuration
st.set_page_config(
    page_title="CityPulse Annotation Tool",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}
.stButton > button {
    width: 100%;
    margin: 0.2rem 0;
}
.change-point {
    background-color: #ff4b4b;
    color: white;
    font-weight: bold;
    padding: 0.5rem;
    border-radius: 0.25rem;
    margin: 0.2rem 0;
}
.no-change {
    background-color: #0ec90e;
    color: white;
    padding: 0.5rem;
    border-radius: 0.25rem;
    margin: 0.2rem 0;
}
.annotation-stats {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

class AnnotationManager:
    """Manage annotations for street view time series"""
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.annotations_file = os.path.join(output_dir, "annotations.json")
        self.progress_file = os.path.join(output_dir, "progress.json")
        
        # Load existing annotations
        self.annotations = self._load_annotations()
        self.progress = self._load_progress()
        
    def _load_annotations(self) -> Dict:
        """Load existing annotations"""
        if os.path.exists(self.annotations_file):
            with open(self.annotations_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_progress(self) -> Dict:
        """Load annotation progress"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "completed": [],
            "skipped": [],
            "in_progress": None,
            "last_updated": None
        }
    
    def save_annotations(self):
        """Save annotations to file"""
        with open(self.annotations_file, 'w') as f:
            json.dump(self.annotations, f, indent=2)
    
    def save_progress(self):
        """Save progress to file"""
        self.progress["last_updated"] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def add_annotation(self, series_id: str, annotation: Dict):
        """Add annotation for a series"""
        self.annotations[series_id] = {
            **annotation,
            "annotated_at": datetime.now().isoformat(),
            "annotator": st.session_state.get("annotator_name", "unknown")
        }
        
        # Update progress
        if series_id not in self.progress["completed"]:
            self.progress["completed"].append(series_id)
        if series_id in self.progress["skipped"]:
            self.progress["skipped"].remove(series_id)
            
        self.save_annotations()
        self.save_progress()
    
    def skip_series(self, series_id: str, reason: str):
        """Mark series as skipped"""
        if series_id not in self.progress["skipped"]:
            self.progress["skipped"].append(series_id)
        
        self.annotations[series_id] = {
            "skipped": True,
            "reason": reason,
            "annotated_at": datetime.now().isoformat(),
            "annotator": st.session_state.get("annotator_name", "unknown")
        }
        
        self.save_annotations()
        self.save_progress()
    
    def get_statistics(self) -> Dict:
        """Get annotation statistics"""
        stats = {
            "total_annotated": len(self.progress["completed"]),
            "total_skipped": len(self.progress["skipped"]),
            "total_changes": sum(
                len(ann.get("change_points", [])) 
                for ann in self.annotations.values() 
                if not ann.get("skipped", False)
            ),
            "annotators": list(set(
                ann.get("annotator", "unknown") 
                for ann in self.annotations.values()
            ))
        }
        
        # Changes by year
        changes_by_year = defaultdict(int)
        for ann in self.annotations.values():
            if not ann.get("skipped", False):
                for year in ann.get("change_points", []):
                    changes_by_year[year] += 1
        
        stats["changes_by_year"] = dict(changes_by_year)
        
        return stats

class ImageViewer:
    """Enhanced image viewer with comparison tools"""
    
    @staticmethod
    def load_image(image_path: str, size: Tuple[int, int] = (640, 640)) -> Optional[Image.Image]:
        """Load and resize image"""
        if os.path.exists(image_path):
            img = Image.open(image_path)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            return img
        else:
            # Return placeholder
            return Image.new('RGB', size, color='lightgray')
    
    @staticmethod
    def create_comparison_view(img1: Image.Image, img2: Image.Image) -> Image.Image:
        """Create side-by-side comparison view"""
        width = img1.width + img2.width + 20
        height = max(img1.height, img2.height)
        
        combined = Image.new('RGB', (width, height), color='white')
        combined.paste(img1, (0, 0))
        combined.paste(img2, (img1.width + 20, 0))
        
        return combined
    
    @staticmethod
    def create_difference_map(img1: Image.Image, img2: Image.Image) -> np.ndarray:
        """Create difference heatmap between images"""
        # Convert to grayscale arrays
        arr1 = np.array(img1.convert('L')).astype(float)
        arr2 = np.array(img2.convert('L')).astype(float)
        
        # Ensure same size
        if arr1.shape != arr2.shape:
            min_h = min(arr1.shape[0], arr2.shape[0])
            min_w = min(arr1.shape[1], arr2.shape[1])
            arr1 = arr1[:min_h, :min_w]
            arr2 = arr2[:min_h, :min_w]
        
        # Calculate difference
        diff = np.abs(arr1 - arr2)
        
        return diff

def load_time_series_data(data_dir: str) -> List[Dict]:
    """Load all time series data from directory"""
    all_data = []
    
    # Look for JSON files
    for filename in os.listdir(data_dir):
        if filename.endswith('_timeseries.json'):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
    
    return all_data

def main():
    """Main annotation interface"""
    
    # Title
    st.title("🏙️ CityPulse Change Point Annotation Tool")
    st.markdown("Label urban changes in street view time series data")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Annotator name
        annotator_name = st.text_input(
            "Your Name",
            value=st.session_state.get("annotator_name", ""),
            help="Enter your name for tracking annotations"
        )
        if annotator_name:
            st.session_state["annotator_name"] = annotator_name
        
        # Data directory
        data_dir = st.text_input(
            "Data Directory",
            value="./data",
            help="Directory containing time series JSON files"
        )
        
        # Output directory
        output_dir = st.text_input(
            "Output Directory",
            value="./annotations",
            help="Directory to save annotations"
        )
        
        # Load data button
        if st.button("Load Data", type="primary"):
            if os.path.exists(data_dir):
                st.session_state["data"] = load_time_series_data(data_dir)
                st.session_state["manager"] = AnnotationManager(data_dir, output_dir)
                st.session_state["current_index"] = 0
                st.success(f"Loaded {len(st.session_state['data'])} time series")
            else:
                st.error(f"Data directory not found: {data_dir}")
        
        # Statistics
        if "manager" in st.session_state:
            st.header("Progress")
            stats = st.session_state["manager"].get_statistics()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Annotated", stats["total_annotated"])
                st.metric("Total Changes", stats["total_changes"])
            with col2:
                st.metric("Skipped", stats["total_skipped"])
                st.metric("Annotators", len(stats["annotators"]))
            
            # Progress bar
            if "data" in st.session_state:
                total = len(st.session_state["data"])
                completed = stats["total_annotated"] + stats["total_skipped"]
                progress = completed / total if total > 0 else 0
                st.progress(progress)
                st.text(f"{completed}/{total} completed ({progress:.1%})")
    
    # Main annotation interface
    if "data" not in st.session_state:
        st.info("👈 Please configure and load data from the sidebar")
        return
    
    if not st.session_state.get("annotator_name"):
        st.warning("⚠️ Please enter your name in the sidebar before annotating")
        return
    
    # Get current series
    data = st.session_state["data"]
    current_index = st.session_state.get("current_index", 0)
    
    if current_index >= len(data):
        st.success("🎉 All series have been annotated!")
        return
    
    series = data[current_index]
    manager = st.session_state["manager"]
    
    # Skip if already annotated
    if series["location_id"] in manager.annotations:
        st.session_state["current_index"] += 1
        st.rerun()
    
    # Series information
    st.header(f"Series: {series['location_id']}")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.text(f"City: {series.get('city', 'Unknown')}")
    with col2:
        st.text(f"Images: {len(series['images'])}")
    with col3:
        years = [img['year'] for img in series['images']]
        st.text(f"Years: {min(years)} - {max(years)}")
    
    # Image gallery
    st.subheader("Time Series Images")
    
    # Initialize session state for annotations
    if "current_annotations" not in st.session_state:
        st.session_state["current_annotations"] = []
    
    # Display images in grid
    images = series["images"]
    n_cols = min(5, len(images))
    
    # Create columns for images
    cols = st.columns(n_cols)
    
    for i, img_data in enumerate(images):
        col_idx = i % n_cols
        
        with cols[col_idx]:
            # Load image
            img_path = img_data.get("image_path", "")
            img = ImageViewer.load_image(img_path, size=(200, 200))
            
            # Display image
            st.image(img, caption=f"{img_data['year']}", use_column_width=True)
            
            # Change point button
            is_change = img_data['year'] in st.session_state["current_annotations"]
            
            if st.button(
                "🔴 Change Point" if is_change else "⚪ Mark Change",
                key=f"btn_{i}",
                help="Click to toggle change point"
            ):
                if is_change:
                    st.session_state["current_annotations"].remove(img_data['year'])
                else:
                    st.session_state["current_annotations"].append(img_data['year'])
                st.rerun()
    
    # Comparison view
    st.subheader("Image Comparison")
    
    col1, col2 = st.columns(2)
    with col1:
        img1_idx = st.selectbox(
            "First Image",
            range(len(images)),
            format_func=lambda x: f"{images[x]['year']}"
        )
    with col2:
        img2_idx = st.selectbox(
            "Second Image",
            range(len(images)),
            index=min(img1_idx + 1, len(images) - 1),
            format_func=lambda x: f"{images[x]['year']}"
        )
    
    if img1_idx != img2_idx:
        # Load images for comparison
        img1 = ImageViewer.load_image(images[img1_idx].get("image_path", ""))
        img2 = ImageViewer.load_image(images[img2_idx].get("image_path", ""))
        
        # Show comparison
        tab1, tab2, tab3 = st.tabs(["Side by Side", "Overlay", "Difference"])
        
        with tab1:
            comparison = ImageViewer.create_comparison_view(img1, img2)
            st.image(comparison, caption=f"{images[img1_idx]['year']} vs {images[img2_idx]['year']}")
        
        with tab2:
            # Slider for overlay
            alpha = st.slider("Opacity", 0.0, 1.0, 0.5)
            overlay = Image.blend(img1, img2, alpha)
            st.image(overlay, caption="Overlay View")
        
        with tab3:
            # Difference heatmap
            diff = ImageViewer.create_difference_map(img1, img2)
            fig = px.imshow(diff, color_continuous_scale='hot', title="Difference Heatmap")
            st.plotly_chart(fig, use_container_width=True)
    
    # Annotation summary
    st.subheader("Annotation Summary")
    
    if st.session_state["current_annotations"]:
        st.markdown(
            f"<div class='change-point'>Change Points: {', '.join(map(str, sorted(st.session_state['current_annotations'])))}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='no-change'>No changes detected in this series</div>",
            unsafe_allow_html=True
        )
    
    # Notes
    notes = st.text_area(
        "Additional Notes",
        placeholder="Add any observations about the changes...",
        help="Describe the type of changes observed (construction, demolition, renovation, etc.)"
    )
    
    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("✅ Save & Next", type="primary"):
            # Save annotation
            annotation = {
                "change_points": sorted(st.session_state["current_annotations"]),
                "notes": notes,
                "images": series["images"]
            }
            manager.add_annotation(series["location_id"], annotation)
            
            # Reset and move to next
            st.session_state["current_annotations"] = []
            st.session_state["current_index"] += 1
            st.success("Annotation saved!")
            st.rerun()
    
    with col2:
        if st.button("⏭️ Skip", type="secondary"):
            reason = st.text_input("Skip reason", key="skip_reason")
            if reason:
                manager.skip_series(series["location_id"], reason)
                st.session_state["current_index"] += 1
                st.rerun()
    
    with col3:
        if st.button("⬅️ Previous"):
            if current_index > 0:
                st.session_state["current_index"] -= 1
                st.session_state["current_annotations"] = []
                st.rerun()
    
    with col4:
        # Jump to specific series
        jump_to = st.number_input(
            "Jump to",
            min_value=1,
            max_value=len(data),
            value=current_index + 1,
            key="jump_to"
        )
        if st.button("Go"):
            st.session_state["current_index"] = jump_to - 1
            st.session_state["current_annotations"] = []
            st.rerun()
    
    # Keyboard shortcuts help
    with st.expander("⌨️ Keyboard Shortcuts"):
        st.text("""
        Coming soon:
        - Space: Toggle change point for current image
        - Enter: Save and next
        - S: Skip current series
        - Left/Right arrows: Navigate images
        """)

def export_annotations():
    """Export annotations in various formats"""
    st.header("📤 Export Annotations")
    
    if "manager" not in st.session_state:
        st.warning("No annotations to export")
        return
    
    manager = st.session_state["manager"]
    annotations = manager.annotations
    
    if not annotations:
        st.warning("No annotations to export")
        return
    
    # Export format
    format_option = st.selectbox(
        "Export Format",
        ["JSON", "CSV", "COCO", "Custom"]
    )
    
    if format_option == "JSON":
        # Standard JSON export
        json_str = json.dumps(annotations, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"citypulse_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    elif format_option == "CSV":
        # Convert to CSV format
        rows = []
        for series_id, ann in annotations.items():
            if not ann.get("skipped", False):
                for year in ann.get("change_points", []):
                    rows.append({
                        "series_id": series_id,
                        "change_year": year,
                        "annotator": ann.get("annotator", "unknown"),
                        "annotated_at": ann.get("annotated_at", ""),
                        "notes": ann.get("notes", "")
                    })
        
        df = pd.DataFrame(rows)
        csv_str = df.to_csv(index=False)
        
        st.download_button(
            label="Download CSV",
            data=csv_str,
            file_name=f"citypulse_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Preview
    st.subheader("Preview")
    st.json(list(annotations.items())[:5])

# Run the app
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="CityPulse Annotation Tool")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--output-dir", default="./annotations", help="Output directory")
    
    args = parser.parse_args()
    
    # Set default values in session state
    if "data_dir" not in st.session_state:
        st.session_state["data_dir"] = args.data_dir
    if "output_dir" not in st.session_state:
        st.session_state["output_dir"] = args.output_dir
    
    # Run main interface
    main()
    
    # Add export section at the bottom
    with st.expander("📤 Export Options"):
        export_annotations()
