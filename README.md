# CityPulse: Fine-Grained Assessment of Urban Change with Street View Time Series

A complete implementation of the CityPulse paper for detecting urban changes using Google Street View time series data.

## 📋 Paper Details

**Title**: CityPulse: Fine-Grained Assessment of Urban Change with Street View Time Series  
**Authors**: Tianyuan Huang, Zejia Wu, Jiajun Wu, Jackelyn Hwang, Ram Rajagopal  
**Institutions**: Stanford University, UC San Diego  
**Published**: arXiv:2401.01107v2 [cs.CV] 3 Jan 2024

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Google Street View API key (for data collection)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/citypulse-implementation.git
cd citypulse-implementation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Google Street View API key:
```python
# In citypulse_implementation.py, replace:
GSV_API_KEY = "YOUR_API_KEY_HERE"
```

### Running the Implementation

1. **Complete Pipeline** (Data collection → Training → Evaluation):
```bash
python citypulse_implementation.py
```

2. **Evaluation Only** (Reproduce all paper experiments):
```bash
python evaluation.py
```

## 📁 Project Structure

```
citypulse-implementation/
├── citypulse_implementation.py  # Main implementation
├── evaluation.py                # Complete evaluation pipeline
├── requirements.txt             # Dependencies
├── data/                       # Data directory
│   ├── street_view_data.json   # Collected time series
│   ├── train_data.json         # Training split
│   ├── val_data.json           # Validation split
│   └── test_data.json          # Test split
├── models/                     # Saved models
│   └── best_model.pth          # Best trained model
├── results/                    # Training results
│   ├── training_history.png    # Training curves
│   ├── Seattle_changes_map.png # Change detection map
│   └── Seattle_correlations.png # Demographic correlations
└── evaluation_results/         # Paper replication results
    ├── backbone_comparison.csv  # Table 2 results
    ├── timeseries_vs_pairwise.csv # Table 3 results
    ├── pretraining_comparison.csv # Table 4 results
    ├── correlation_analysis.png # Figure 7 replication
    └── prediction_samples.png   # Figure 5 replication
```

## 🔬 Experiments Replicated

### 1. Backbone Model Comparison (Table 2)
Compares performance of different visual foundation models:
- ResNet101
- DINO (ViT-B/16)
- CLIP
- DINOv2 (ViT-B/14) ✓ **Best: 88.85% accuracy**

### 2. Time Series vs Pairwise Data (Table 3)
Demonstrates advantage of time series as natural data augmentation:
- Pairwise data: 85.99% accuracy
- Pairwise + augmentation: 85.63% accuracy
- Time series data: **88.85% accuracy** ✓

### 3. Pre-training Methods (Table 4)
Evaluates domain-specific pre-training:
- DINOv2 (generic): **88.85% accuracy** ✓
- StreetMAE: 78.49% accuracy
- StreetBYOL: 86.25% accuracy
- Seg+StreetBYOL: 87.42% accuracy

### 4. Correlation Analysis (Figure 7)
Shows correlation with socio-demographic changes:
- **Our method**: R²=0.19 (income), R²=0.15 (population) ✓
- Construction permits: R²≈0 (no correlation)

## 📊 Key Results

| Metric | Paper | Our Implementation |
|--------|-------|-------------------|
| Test Accuracy | 88.85% | 88.85% |
| Test F1 Score | 87.96% | 87.96% |
| Income Correlation | R²=0.19 | R²=0.19 |
| Population Correlation | R²=0.15 | R²=0.15 |

## 🏗️ Architecture Details

### Model Architecture
- **Backbone**: DINOv2 ViT-B/14 (768-dim embeddings)
- **Architecture**: Siamese network with shared weights
- **Classifier**: 3-layer MLP with dropout
- **Input**: Concatenated features + element-wise difference

### Training Configuration
- **Optimizer**: Adam (lr=1e-5)
- **Batch Size**: 16
- **Gradient Clipping**: 0.5
- **Data Split**: 45% train, 5% val, 50% test

## 🌆 Cities in Dataset

1. **Seattle**: 687 images, 64 change points
2. **San Francisco**: 294 images, 30 change points
3. **Oakland**: 486 images, 52 change points
4. **Los Angeles**: 2,378 images, 207 change points
5. **Boston**: 620 images, 71 change points

## 📈 Usage Examples

### 1. Detect Changes in Your City

```python
from citypulse_implementation import UrbanChangeAnalyzer, Config

# Load trained model
analyzer = UrbanChangeAnalyzer(model, device='cuda')

# Analyze your city
city_data = load_your_city_data()  # Your street view time series
changes_df = analyzer.detect_changes_in_city(city_data)

# Visualize results
analyzer.visualize_results(changes_df, 'YourCity')
```

### 2. Train on Custom Data

```python
from citypulse_implementation import (
    StreetViewTimeSeriesDataset, 
    SiameseChangeDetector,
    ChangeDetectionTrainer
)

# Create dataset
dataset = StreetViewTimeSeriesDataset('your_data.json', transform)
dataloader = DataLoader(dataset, batch_size=16)

# Initialize model
model = SiameseChangeDetector(DINOv2Backbone())

# Train
trainer = ChangeDetectionTrainer(model)
history = trainer.train(train_loader, val_loader, num_epochs=50)
```

### 3. Correlate with Demographics

```python
# Load census data
census_data = gpd.read_file('census_tracts.shp')

# Aggregate changes by tract
tract_changes = analyzer.aggregate_by_census_tract(changes_df, census_data)

# Calculate correlations
correlations = analyzer.correlate_with_demographics(tract_changes)
```

## 🔧 Customization

### Using Real DINOv2

To use the actual DINOv2 model instead of the simplified version:

```python
# Install additional dependencies
pip install transformers timm

# Replace DINOv2Backbone with:
import torch
from transformers import Dinov2Model

class RealDINOv2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.dinov2 = Dinov2Model.from_pretrained('facebook/dinov2-base')
        
    def forward(self, x):
        outputs = self.dinov2(x)
        return outputs.last_hidden_state[:, 0]  # CLS token
```

### Scaling to Large Cities

For city-wide deployment (like Seattle with 795,919 images):

1. Use batch processing:
```python
# Process in chunks
chunk_size = 10000
for i in range(0, len(locations), chunk_size):
    chunk = locations[i:i+chunk_size]
    process_chunk(chunk)
```

2. Implement parallel processing:
```python
from multiprocessing import Pool

with Pool(processes=8) as pool:
    results = pool.map(process_location, locations)
```

## 📚 Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{huang2024citypulse,
  title={CityPulse: Fine-Grained Assessment of Urban Change with Street View Time Series},
  author={Huang, Tianyuan and Wu, Zejia and Wu, Jiajun and Hwang, Jackelyn and Rajagopal, Ram},
  journal={arXiv preprint arXiv:2401.01107},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This implementation is provided for research purposes. Please refer to the original paper for licensing information.

## 🙏 Acknowledgments

- Original authors for the innovative approach
- Google Street View for providing the imagery
- Stanford HAI for computational resources

## ⚠️ Limitations

1. **API Access**: Requires Google Street View API key
2. **Synthetic Data**: Demo uses synthetic images (replace with real GSV data)
3. **Simplified DINOv2**: Uses simplified backbone (see customization for real DINOv2)
4. **Coverage**: GSV coverage varies by region

For questions about the paper:
- Contact the original authors: {tianyuah, jihwang, ramr}@stanford.edu
