import os
import json
import argparse
import torch
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torchvision.transforms import Resize
from transformers import AutoModel
from tqdm import tqdm
import torch.nn.functional as F

#command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Neighborhood Gradient Descent for MoAI")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference JSON file")
    parser.add_argument("--eval", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--num_neighbors", type=int, default=5, help="Number of neighbors for KNN")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of optimization steps")
    parser.add_argument("--initial_lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-5, help="Final learning rate")
    return parser.parse_args()

class NGDAdaptor:
    def __init__(self, ref_data_path, num_neighbors=3):
        self.num_neighbors = num_neighbors
        self.ref_samples = self.load_reference_data(ref_data_path)
        self.embed_model = self.init_embed_model()
        self.knn_index = self.build_knn_index()
        
        # Initialize MoAI model
        self.moai, self.processor = self.init_moai_model()
        self.frozen_model_parameters()  # Freeze non-routing parameters
        
    def init_embed_model(self):
        """Initialize embedding model with 8-bit quantization"""
        return AutoModel.from_pretrained(
            'nvidia/NV-Embed-v2',
            device_map="auto",
            load_in_8bit=True,
            torch_dtype=torch.float16
        )
    
    def load_reference_data(self, path):
        """Load reference data with precomputed features"""
        with open(path) as f:
            data = json.load(f)
        return data
    
    def build_knn_index(self):
        """Build KNN index using precomputed embeddings"""
        questions = [s['question'] for s in self.ref_samples]
        embeddings = self.compute_embedding(questions)
        return NearestNeighbors(n_neighbors=self.num_neighbors, metric='cosine').fit(embeddings)
    
    def compute_embedding(self, texts):
        """Compute embeddings for a list of texts"""
        with torch.no_grad():
            embeddings = self.embed_model.encode(texts, instruction="Retrieve relevant questions: ")
        return F.normalize(embeddings, p=2, dim=1).cpu().numpy()
    
    def init_moai_model(self):
        """Initialize MoAI model with configurable routing weights"""
        moai, processor, *_ = prepare_moai(
            moai_path='BK-Lee/MoAI-7B',
            bits=4,
            grad_ckpt=False,
            lora=False,
            dtype='fp16'
        )
        return moai, processor
    
    def frozen_model_parameters(self):
        """Freeze all parameters except the six MoAI attention modules"""
        trainable_modules = [
            "moai_CA_img_aux",  # I_AUX
            "moai_CA_img_lang", # I_LANG
            "moai_SA_img",      # I_SELF
            "moai_CA_lang_aux", # L_AUX
            "moai_CA_lang_img", # L_IMG
            "moai_SA_lang"      # L_SELF
        ]
        
        for name, param in self.moai.named_parameters():
            if not any(module in name for module in trainable_modules):
                param.requires_grad = False
    
    def ngd_step(self, test_item, num_steps=10, initial_lr=0.01, final_lr=1e-5):
        """Core NGD optimization step with cosine learning rate decay"""
        # 1. Find neighbors
        distances, indices = self.knn_index.kneighbors(self.compute_embedding([test_item['question']]))
        neighbors = [self.ref_samples[i] for i in indices[0]]
        similarities = 1 - distances[0]
        
        # 2. Prepare routing parameters
        routing_params = [p for p in self.moai.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(routing_params, lr=initial_lr)
        
        # 3. Initialize cosine learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_steps,
            eta_min=final_lr
        )
        
        # 4. Neighborhood gradient descent
        for step in range(num_steps):
            total_loss = 0
            for i, neighbor in enumerate(neighbors):
                inputs = self.prepare_inputs(neighbor)
                outputs = self.moai(**inputs)
                loss = self.compute_loss(outputs, neighbor['ground_truth'])
                total_loss += similarities[i] * loss
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            (total_loss / similarities.sum()).backward()
            optimizer.step()
            scheduler.step()
            
            print(f"Step {step+1}/{num_steps} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # 5. Generate final prediction
        return self.generate_answer(test_item)
    
    def prepare_inputs(self, item):
        """Process image/text inputs for MoAI"""
        image = Resize((490, 490))(Image.open(item['image_path']))
        return self.processor(
            images=[image],
            text=item['question'],
            return_tensors="pt"
        ).to(self.moai.device)
    
    def compute_loss(self, outputs, ground_truth):
        """Calculate cross entropy loss"""
        answer_ids = self.processor(
            ground_truth, 
            return_tensors="pt",
            padding=True,
            truncation=True
        )["input_ids"].to(outputs.logits.device)
        
        return F.cross_entropy(
            outputs.logits[:, -answer_ids.size(1):, :].flatten(0, 1),
            answer_ids.flatten(),
            ignore_index=self.processor.tokenizer.pad_token_id
        )
    
    def generate_answer(self, item):
        """Generate answer with optimized routing weights"""
        inputs = self.prepare_inputs(item)
        generate_ids = self.moai.generate(
            **inputs,
            do_sample=True,
            temperature=0.05,
            max_new_tokens=256
        )
        return self.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]

def main():
    args = parse_args()
    
    # Initialize adaptor with reference data
    adaptor = NGDAdaptor(args.reference, num_neighbors=args.num_neighbors)
    
    # Load evaluation data
    with open(args.eval) as f:
        eval_data = json.load(f)['samples']
    
    # Run evaluation
    results = []
    for item in tqdm(eval_data):
        optimized_answer = adaptor.ngd_step(
            item,
            num_steps=args.num_steps,
            initial_lr=args.initial_lr,
            final_lr=args.final_lr
        )
        results.append({
            "prediction": optimized_answer,
            "ground_truth": item['correct_answer']
        })
    
    # Calculate accuracy
    accuracy = np.mean([r['prediction'].startswith(r['ground_truth']) for r in results])
    print(f"Final Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
