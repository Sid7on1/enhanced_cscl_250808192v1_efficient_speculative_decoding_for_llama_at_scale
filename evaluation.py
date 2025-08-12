import logging
import numpy as np
import pandas as pd
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import Dict, List, Optional
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import MSELoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import json
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants and configuration
CONFIG_FILE = 'config.json'
MODEL_FILE = 'model.pth'
TOKENIZER_FILE = 'tokenizer.pkl'
METRICS_FILE = 'metrics.json'

class EvaluationMetrics:
    def __init__(self):
        self.metrics = {
            'pearsonr': [],
            'mse': [],
            'rmse': [],
            'accuracy': []
        }

    def update(self, pearsonr_value, mse_value, rmse_value, accuracy_value):
        self.metrics['pearsonr'].append(pearsonr_value)
        self.metrics['mse'].append(mse_value)
        self.metrics['rmse'].append(rmse_value)
        self.metrics['accuracy'].append(accuracy_value)

    def save(self, file_path):
        with open(file_path, 'w') as f:
            json.dump(self.metrics, f)

class EvaluationDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }

class EvaluationModel(torch.nn.Module):
    def __init__(self, model_name):
        super(EvaluationModel, self).__init__()
        self.model = LlamaForCausalLM.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

class Evaluation:
    def __init__(self, model_name, config):
        self.model_name = model_name
        self.config = config
        self.metrics = EvaluationMetrics()
        self.model = EvaluationModel(model_name)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        return data

    def preprocess_data(self, data):
        data['label'] = data['label'].astype(float)
        return data

    def create_dataset(self, data):
        dataset = EvaluationDataset(data, self.tokenizer)
        return dataset

    def train(self, dataset, batch_size):
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = Adam(self.model.parameters(), lr=self.config['lr'])
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config['epochs'])
        loss_fn = MSELoss()
        for epoch in range(self.config['epochs']):
            self.model.train()
            total_loss = 0
            with tqdm(train_loader, desc=f'Epoch {epoch+1}') as pbar:
                for batch in pbar:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(input_ids, attention_mask)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': f'{total_loss / (epoch + 1):.4f}'})
            scheduler.step()
        self.model.eval()

    def evaluate(self, dataset):
        self.model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for batch in dataset:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids, attention_mask)
                predictions.extend(outputs.cpu().numpy())
                labels.extend(batch['label'].cpu().numpy())
        pearsonr_value, _ = pearsonr(predictions, labels)
        mse_value = mean_squared_error(predictions, labels)
        rmse_value = np.sqrt(mse_value)
        accuracy_value = np.mean(np.abs(predictions - labels) < 1e-6)
        self.metrics.update(pearsonr_value, mse_value, rmse_value, accuracy_value)

    def save_metrics(self, file_path):
        self.metrics.save(file_path)

    def run(self, file_path, batch_size):
        data = self.load_data(file_path)
        data = self.preprocess_data(data)
        dataset = self.create_dataset(data)
        self.train(dataset, batch_size)
        self.evaluate(dataset)
        self.save_metrics(METRICS_FILE)

def main():
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    evaluation = Evaluation('llama-base', config)
    evaluation.run('data.csv', 32)

if __name__ == '__main__':
    main()