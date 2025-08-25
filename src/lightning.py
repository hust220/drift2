import pytorch_lightning as pl
import torch

from src import const
from src.gnn import GNN
from src.const import TORCH_INT
from src.pocket_ligand_dataset import (
    PocketLigandDataset, get_pocket_ligand_dataloader, collate_pocket_ligand
)
from typing import Dict, List, Optional
from torch.nn import functional as F

def get_activation(activation):
    if activation == 'silu':
        return torch.nn.SiLU()
    else:
        raise Exception("activation fn not supported yet. Add it here.")

class Drift2(pl.LightningModule):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    starting_epoch = None
    metrics: Dict[str, List[float]] = {}

    FRAMES = 100

    def __init__(
        self,
        hidden_nf,
        activation='silu', n_layers=3, sin_embedding=True, normalization_factor=1, aggregation_method='sum',
        batch_size=2, lr=1e-4, torch_device='cpu', test_epochs=1, n_stability_samples=1,
        log_iterations=None, samples_dir=None, data_augmentation=False, table_name='pdbbind_dataset',
    ):
        super(Drift2, self).__init__()

        self.save_hyperparameters()
        self.batch_size = batch_size
        self.lr = lr
        self.torch_device = torch_device
        self.test_epochs = test_epochs
        self.n_stability_samples = n_stability_samples
        self.log_iterations = log_iterations
        self.samples_dir = samples_dir
        self.data_augmentation = data_augmentation
        self.table_name = table_name

        self.in_node_nf = const.N_RESIDUE_TYPES + const.N_ATOM_TYPES  # 氨基酸和原子的one-hot编码
        self.in_edge_nf = 1 + 1 + 1 + 1 + const.N_RDBOND_TYPES  # distance, is_backbone, is_pocket_mol, is_mol_mol, bond_features
        self.hidden_nf = hidden_nf
        self.out_node_nf = hidden_nf  # 输出节点特征
        self.out_edge_nf = 1  # 边特征输出

        self.validation_step_outputs = []
        self.training_step_outputs = []
        self.test_step_outputs = []

        if type(activation) is str:
            activation = get_activation(activation)

        # GNN for graph representation
        self.gnn = GNN(
            in_node_nf=self.in_node_nf,
            in_edge_nf=self.in_edge_nf,
            hidden_nf=hidden_nf,
            out_node_nf=self.out_node_nf,
            out_edge_nf=self.out_edge_nf,
            n_layers=n_layers,
            sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
        )
        
        # Final layers to produce scalar output
        self.graph_pooling = torch.nn.Sequential(
            torch.nn.Linear(hidden_nf, hidden_nf),
            activation,
            torch.nn.Linear(hidden_nf, 1)
        )

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = PocketLigandDataset(
            device=self.torch_device,
            split='train',
            table_name=self.table_name
        )
        self.val_dataset = PocketLigandDataset(
            device=self.torch_device,
            split='val',
            table_name=self.table_name
        )

    def train_dataloader(self):
        return get_pocket_ligand_dataloader(self.train_dataset, self.batch_size, shuffle=True)

    def val_dataloader(self):
        return get_pocket_ligand_dataloader(self.val_dataset, self.batch_size, shuffle=False)

    def test_dataloader(self):
        return get_pocket_ligand_dataloader(self.test_dataset, self.batch_size, shuffle=False)

    def forward(self, data):
        """
        Forward pass for a single graph (either positive or negative)
        Returns a scalar score for the pocket-ligand interaction
        """
        h = data['one_hot']
        edge_index = data['edge_index'].to(TORCH_INT)
        edge_attr = data['edge_attr']
        node_mask = data['node_mask']
        edge_mask = data['edge_mask']

        # Get node representations from GNN
        node_features, _ = self.gnn.forward(
            h=h,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_mask=node_mask,
            edge_mask=edge_mask,
        )

        # Global graph pooling (mean pooling over all nodes)
        # node_features: [batch, max_nodes, hidden_nf]
        # node_mask: [batch, max_nodes, 1]
        masked_features = node_features * node_mask
        graph_representation = masked_features.sum(dim=1) / (node_mask.sum(dim=1) + 1e-8)
        
        # Get scalar score
        score = self.graph_pooling(graph_representation).squeeze(-1)  # [batch]
        
        return score
    
    def loss_fn(self, positive_scores, negative_scores, margin=1.0):
        """
        Contrastive loss: positive scores should be higher than negative scores
        positive_scores: [batch] - scores for positive pairs
        negative_scores: [batch] - scores for negative pairs
        """
        # Margin ranking loss: max(0, margin - (positive - negative))
        loss = F.margin_ranking_loss(
            positive_scores, 
            negative_scores, 
            torch.ones_like(positive_scores),  # target: positive should be > negative
            margin=margin,
            reduction='mean'
        )
        return loss

    def training_step(self, data, *args):
        # data contains 'positive' and 'negative' batches
        positive_data = data['positive']
        negative_data = data['negative']
        
        # Get scores for positive and negative pairs
        positive_scores = self.forward(positive_data)
        negative_scores = self.forward(negative_data)
        
        # Compute contrastive loss
        loss = self.loss_fn(positive_scores, negative_scores)
        
        # Calculate accuracy (how often positive > negative)
        accuracy = (positive_scores > negative_scores).float().mean()
        
        training_metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'positive_score_mean': positive_scores.mean(),
            'negative_score_mean': negative_scores.mean(),
        }
        
        if self.log_iterations is not None and self.global_step % self.log_iterations == 0:
            for metric_name, metric in training_metrics.items():
                self.metrics.setdefault(f'{metric_name}/train', []).append(metric)
                # Only show loss and accuracy in progress bar
                show_in_progbar = metric_name in ['loss', 'accuracy']
                self.log(f'{metric_name}/train', metric, prog_bar=show_in_progbar)

        # Return only loss to avoid Lightning's auto-logging
        return loss

    def validation_step(self, data, *args):
        # data contains 'positive' and 'negative' batches
        positive_data = data['positive']
        negative_data = data['negative']
        
        # Get scores for positive and negative pairs
        positive_scores = self.forward(positive_data)
        negative_scores = self.forward(negative_data)
        
        # Compute contrastive loss
        loss = self.loss_fn(positive_scores, negative_scores)
        
        # Calculate accuracy (how often positive > negative)
        accuracy = (positive_scores > negative_scores).float().mean()
        
        rt = {
            'loss': loss,
            'accuracy': accuracy,
            'positive_score_mean': positive_scores.mean(),
            'negative_score_mean': negative_scores.mean(),
        }
        self.validation_step_outputs.append(rt)
        return loss  # Return only loss to avoid auto-logging

    def test_step(self, data, *args):
        # data contains 'positive' and 'negative' batches
        positive_data = data['positive']
        negative_data = data['negative']
        
        # Get scores for positive and negative pairs
        positive_scores = self.forward(positive_data)
        negative_scores = self.forward(negative_data)
        
        # Compute contrastive loss
        loss = self.loss_fn(positive_scores, negative_scores)
        
        # Calculate accuracy (how often positive > negative)
        accuracy = (positive_scores > negative_scores).float().mean()
        
        rt = {
            'loss': loss,
            'accuracy': accuracy,
            'positive_score_mean': positive_scores.mean(),
            'negative_score_mean': negative_scores.mean(),
        }
        self.test_step_outputs.append(rt)
        return loss  # Return only loss to avoid auto-logging

    def on_validation_epoch_end(self):
        for metric in self.validation_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(self.validation_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/val', []).append(avg_metric)
            # Only show loss and accuracy in progress bar
            show_in_progbar = metric in ['loss', 'accuracy']
            self.log(f'{metric}/val', avg_metric, prog_bar=show_in_progbar)

        self.validation_step_outputs = []

    def on_test_epoch_end(self):
        for metric in self.test_step_outputs[0].keys():
            avg_metric = self.aggregate_metric(self.test_step_outputs, metric)
            self.metrics.setdefault(f'{metric}/test', []).append(avg_metric)
            # Only show loss and accuracy in progress bar
            show_in_progbar = metric in ['loss', 'accuracy']
            self.log(f'{metric}/test', avg_metric, prog_bar=show_in_progbar)

        self.test_step_outputs = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.gnn.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12)

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        return torch.tensor([out[metric] for out in step_outputs]).mean()
