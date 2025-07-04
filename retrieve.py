import torch
import open_clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
import torchvision.transforms as transforms
from tqdm import tqdm

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the fine-tuned model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained=False)
model.load_state_dict(torch.load('clip_fashion_final.pt'))
model = model.to(device)
model.eval()

tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')

# Load dataset for image retrieval
dataset = load_dataset('ceyda/fashion-products-small')
fashion_data = dataset['train']

# Create image embeddings database
def create_image_embeddings(data, num_samples=1000):
    """Create embeddings for a sample of images from the dataset"""
    print(f"Creating embeddings for {num_samples} images...")
    
    # Select random samples
    if num_samples < len(data):
        indices = np.random.choice(len(data), num_samples, replace=False).tolist()  # Convert to Python list
    else:
        indices = list(range(len(data)))  # Ensure Python list
    
    image_embeddings = []
    image_ids = []
    image_metadata = []
    
    with torch.no_grad():
        for idx in tqdm(indices):
            # Convert index to int to avoid numpy.int64 issues
            idx = int(idx)
            item = data[idx]
            image = preprocess(item['image']).unsqueeze(0).to(device)
            
            # Get image embedding
            embedding = model.encode_image(image)
            embedding = embedding / embedding.norm(dim=1, keepdim=True)
            
            image_embeddings.append(embedding.cpu())
            image_ids.append(idx)
            
            # Store metadata for display
            metadata = {
                'id': idx,
                'gender': item['gender'],
                'masterCategory': item['masterCategory'],
                'subCategory': item['subCategory']
            }
            image_metadata.append(metadata)
    
    return torch.cat(image_embeddings), image_ids, image_metadata

# Function to retrieve images based on text query
def retrieve_images_by_text(text_query, embeddings, image_ids, metadata, data, top_k=5):
    """Retrieve top K images matching the text query"""
    # Tokenize and encode the text
    with torch.no_grad():
        text_tokens = tokenizer([text_query]).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
    
    # Calculate similarities with all images
    similarities = (100.0 * embeddings @ text_features.T.cpu()).squeeze(1)
    
    # Get top K matches
    values, indices = similarities.topk(top_k)
    
    top_images = []
    top_similarities = []
    top_metadata = []
    
    for i, idx in enumerate(indices):
        img_id = image_ids[idx]
        top_images.append(data[img_id]['image'])
        top_similarities.append(values[i].item())
        top_metadata.append(metadata[idx])
    
    return top_images, top_similarities, top_metadata

# Function to retrieve similar images
def retrieve_similar_images(image_id, embeddings, image_ids, metadata, data, top_k=5):
    """Retrieve top K images similar to the given image"""
    # Ensure integer index
    image_id = int(image_id)
    
    # Get the query image embedding
    with torch.no_grad():
        query_image = preprocess(data[image_id]['image']).unsqueeze(0).to(device)
        query_embedding = model.encode_image(query_image)
        query_embedding = query_embedding / query_embedding.norm(dim=1, keepdim=True)
    
    # Calculate similarities with all images
    similarities = (100.0 * embeddings @ query_embedding.T.cpu()).squeeze(1)
    
    # Get top K+1 matches (including the query image itself)
    values, indices = similarities.topk(top_k + 1)
    
    # Skip the first result if it's the query image itself
    if image_ids[indices[0]] == image_id:
        values = values[1:]
        indices = indices[1:]
    else:
        values = values[:top_k]
        indices = indices[:top_k]
    
    top_images = []
    top_similarities = []
    top_metadata = []
    
    for i, idx in enumerate(indices):
        img_id = image_ids[idx]
        top_images.append(data[img_id]['image'])
        top_similarities.append(values[i].item())
        top_metadata.append(metadata[idx])
    
    return top_images, top_similarities, top_metadata

# Visualize results
def visualize_results(query, images, similarities, metadata, title):
    """Display query results with images and metadata"""
    n = len(images)
    plt.figure(figsize=(15, 3*n))
    
    plt.suptitle(title, fontsize=16)
    
    for i, (img, sim, meta) in enumerate(zip(images, similarities, metadata)):
        plt.subplot(n, 1, i+1)
        plt.imshow(img)
        plt.title(f"Similarity: {sim:.2f}% - {meta['gender']} {meta['masterCategory']}, {meta['subCategory']}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(f"retrieval_results_{query.replace(' ', '_')[:30]}.png")
    plt.show()

# Main execution
if __name__ == "__main__":
    # Create embeddings database
    num_samples = 2000  # Adjust based on your memory constraints
    embeddings, image_ids, metadata = create_image_embeddings(fashion_data, num_samples)
    
    # Example text queries to try
    text_queries = [
        "a photo of Men's formal shirt",
        "a photo of Women's casual dress",
        "a photo of blue jeans",
        "a photo of leather handbag",
        "a photo of sports shoes"
    ]
    
    # Perform text-to-image retrieval for each query
    for query in text_queries:
        images, similarities, meta = retrieve_images_by_text(
            query, embeddings, image_ids, metadata, fashion_data
        )
        visualize_results(query, images, similarities, meta, 
                          f"Images matching: '{query}'")
    
    # Example of image-to-image similarity search
    # Pick a random image as query
    query_id = int(np.random.choice(image_ids))  # Convert to int
    query_item = fashion_data[query_id]
    
    print(f"Finding similar items to: {query_item['gender']} {query_item['masterCategory']}, {query_item['subCategory']}")
    
    similar_images, similarities, meta = retrieve_similar_images(
        query_id, embeddings, image_ids, metadata, fashion_data
    )
    
    # Show query image with similar items
    all_images = [query_item['image']] + similar_images
    all_similarities = [100.0] + similarities
    all_metadata = [{
        'gender': query_item['gender'],
        'masterCategory': query_item['masterCategory'],
        'subCategory': query_item['subCategory']
    }] + meta
    
    visualize_results("similar_items", all_images, all_similarities, all_metadata,
                     f"Query image (top) and similar fashion items")
