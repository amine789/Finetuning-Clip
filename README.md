# CLIP Fashion Fine-tuning and Image Retrieval

This project demonstrates the fine-tuning of OpenAI's CLIP model specifically for fashion domain tasks, along with a comprehensive image retrieval system for fashion products.

## Project Overview

We fine-tuned a CLIP (Contrastive Language-Image Pre-training) model on fashion-specific data to improve its understanding of fashion terminology and product descriptions. The project consists of two main components:

1. **CLIP Model Fine-tuning** on fashion datasets
2. **Image Retrieval System** using the fine-tuned model

## Architecture

### Base Model
- **Model**: ViT-B-32-quickgelu
- **Framework**: OpenCLIP
- **Base Weights**: OpenAI's pre-trained CLIP

### Fine-tuning Details
- **Dataset**: fashion-products-small (42,700 images)
- **Batch Size**: 4 (optimized for memory constraints)
- **Learning Rate**: 1e-6
- **Epochs**: 3
- **Optimizer**: AdamW with weight decay
- **Loss Function**: Bidirectional contrastive loss (image-to-text and text-to-image)

## Dataset Structure

The fashion dataset includes:
- Product images (224x224 pixels)
- Text descriptions including:
  - Gender category
  - Master category (e.g., Apparel, Footwear)
  - Sub-category (e.g., Topwear, Bottomwear)

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/clip-fashion-retrieval.git
cd clip-fashion-retrieval

# Create and activate virtual environment
python -m venv clip_env
source clip_env/bin/activate  # On Windows: clip_env\Scripts\activate

# Install dependencies
pip install open_clip_torch transformers datasets pillow torch torchvision
pip install matplotlib numpy scikit-learn tqdm
```

## Usage

### 1. Fine-tuning CLIP

```python
from train_clip import train_fashion_clip

# Train the model
model = train_fashion_clip(
    batch_size=4,
    num_epochs=3,
    learning_rate=1e-6,
    subset_size=2000  # Use subset for quicker training
)

# Save the model
torch.save(model.state_dict(), 'clip_fashion_final.pt')
```

### 2. Image Retrieval

```python
from retrieve_images import FashionRetriever

# Initialize retriever with fine-tuned model
retriever = FashionRetriever('clip_fashion_final.pt')

# Perform text-to-image retrieval
results = retriever.search_by_text("Men's casual jacket", top_k=5)

# Perform image-to-image retrieval
similar_items = retriever.search_by_image('path/to/query/image.jpg', top_k=5)
```

## File Structure

```
clip-fashion-retrieval/
│
├── README.md
├── train_clip.py              # Script for fine-tuning CLIP
├── retrieve_images.py         # Image retrieval implementation
├── utils.py                   # Utility functions
├── requirements.txt           # Project dependencies
│
├── models/
│   └── clip_fashion_final.pt  # Fine-tuned model weights
│
├── data/                      # Dataset (not included)
│   └── fashion-products-small/
│
├── results/
│   ├── retrieval_examples/    # Example retrieval results
│   └── visualization/         # Visualization outputs
│
└── checkpoints/              # Training checkpoints
    ├── clip_fashion_epoch1.pt
    ├── clip_fashion_epoch2.pt
    └── clip_fashion_epoch3.pt
```

## Training Results

### Loss Progression
- Epoch 1: 0.3375
- Epoch 2: 0.2715
- Epoch 3: 0.2503

### Model Performance
The fine-tuned model shows improved understanding of:
- Fashion-specific terminology
- Style descriptions
- Product categories and types

## Image Retrieval Features

### 1. Text-to-Image Search
```python
# Find fashion items using natural language
results = retriever.search_by_text("blue denim jacket with silver buttons")
```

### 2. Image-to-Image Search
```python
# Find similar fashion items
similar = retriever.search_by_image("reference_jacket.jpg")
```

### 3. Batch Processing
```python
# Process multiple queries efficiently
queries = ["red dress", "sneakers", "formal suit"]
results = retriever.batch_search(queries)
```

## Results and Examples

### Text Retrieval Examples
- Query: "Men's formal shirt" → Returns business shirts with high accuracy
- Query: "Women's summer dress" → Returns appropriate seasonal dresses
- Query: "Black leather handbag" → Returns matching accessories

### Similarity Search Examples
- Input: Casual t-shirt → Returns similar casual wear
- Input: High heels → Returns similar footwear styles

## Performance Considerations

### Memory Optimization
- Mixed precision training (FP16)
- Gradient checkpointing
- Small batch sizes for GPU memory constraints

### Inference Optimization
- Batch processing for retrieval
- Cached embeddings for large datasets
- Configurable top_k results

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **Slow Retrieval**
   - Pre-compute and cache embeddings
   - Use batch processing
   - Consider using FAISS for large-scale retrieval

3. **Poor Results**
   - Increase training epochs
   - Use larger subset of data
   - Adjust learning rate

## Future Improvements

1. **Dataset Expansion**
   - Include more diverse fashion items
   - Add seasonal and trend-specific data

2. **Model Enhancements**
   - Experiment with larger CLIP variants (ViT-L/14)
   - Implement curriculum learning

3. **Retrieval System**
   - Add filtering by attributes (color, size, etc.)
   - Implement hybrid search (visual + attributes)
   - Add recommendation features

## Citation

If you use this work, please cite:

```bibtex
@misc{clip_fashion_2024,
  title={CLIP Fashion Fine-tuning and Image Retrieval},
  author={Your Name},
  year={2024},
  url={https://github.com/YOUR_USERNAME/clip-fashion-retrieval}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the original CLIP model
- Hugging Face for the datasets library
- The fashion-products-small dataset creators

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com]

---

*Last updated: May 2025*
