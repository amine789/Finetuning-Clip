import os
import torch
import open_clip
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import gc

# Free up cache and collect garbage
torch.cuda.empty_cache()
gc.collect()

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the dataset
dataset = load_dataset('ceyda/fashion-products-small')
train_data = dataset['train']

# Define transforms
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                         (0.26862954, 0.26130258, 0.27577711))
])

# Create custom dataset class with correct subset handling
class FashionDataset(Dataset):
    def __init__(self, dataset, indices=None, transform=None):
        self.dataset = dataset
        self.indices = indices if indices is not None else range(len(dataset))
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the actual index from our subset indices list
        actual_idx = self.indices[idx]
        
        # Get the item using the actual index
        item = self.dataset[actual_idx]
        image = item['image']
        
        # Create text description
        text = f"a photo of {item['gender']} {item['masterCategory']}, {item['subCategory']}"
        
        # Apply transforms if available
        if self.transform:
            image = self.transform(image)
            
        return image, text

# Create subset indices
subset_size = 10000
indices = list(range(min(subset_size, len(train_data))))

# Create dataset with correct subset handling
train_dataset = FashionDataset(train_data, indices=indices, transform=image_transform)
batch_size = 32  # Very small batch size to save memory
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Corrected model loading
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
model = model.to(device)

# IMPORTANT: Freeze most layers, only fine-tune the last few layers
# This helps prevent NaN losses
for name, param in model.named_parameters():
    if 'visual.transformer.resblocks.11' not in name and 'text.transformer.resblocks.11' not in name:
        param.requires_grad = False

# Use a much lower learning rate
lr = 1e-6  # Reduced from 5e-5

# Setup optimizer with gradient clipping
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr,
    betas=(0.9, 0.98),
    eps=1e-6,
    weight_decay=0.01  # Reduced from 0.2
)

# Add gradient clipping
grad_clip_value = 1.0

# Training parameters
num_epochs = 100  # Reduced number of epochs
loss_img_weight = 1.0
loss_txt_weight = 1.0

# Training loop with stability fixes
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, texts) in enumerate(tqdm(train_loader)):
        # Move data to device
        images = images.to(device)
        texts = tokenizer(texts).to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass - avoid mixed precision initially
        image_features, text_features, logit_scale = model(images, texts)
        
        # Numerical stability: limit logit_scale to avoid exponential explosion
        logit_scale = torch.clamp(logit_scale, 0, 4.6052)  # Max e^4.6052 ≈ 100
        
        # Normalized features
        image_features = image_features / (image_features.norm(dim=1, keepdim=True) + 1e-8)
        text_features = text_features / (text_features.norm(dim=1, keepdim=True) + 1e-8)
        
        # Cosine similarity as logits
        logit_scale = logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        # Ground-truth: diagonal elements should match
        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
        
        # Compute loss
        loss_img = torch.nn.functional.cross_entropy(logits_per_image, ground_truth)
        loss_txt = torch.nn.functional.cross_entropy(logits_per_text, ground_truth)
        loss = (loss_img_weight * loss_img + loss_txt_weight * loss_txt) / 2
        
        # Check for NaN loss
        if torch.isnan(loss):
            print(f"NaN loss detected at batch {batch_idx}. Skipping update.")
            continue
        
        # Backward pass
        loss.backward()
        
        # Apply gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        
        # Update weights
        optimizer.step()
        
        # Clear GPU cache periodically
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
        
        total_loss += loss.item()
        
        #if batch_idx % 10 == 0:
            #print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch: {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    # Save checkpoint for each epoch
    

# Save final model
final_model_path = 'clip_fashion_final.pt'
torch.save(model.state_dict(), final_model_path)
print(f"Training completed! Final model saved to {final_model_path}")
