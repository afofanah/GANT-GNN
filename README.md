# GANT-GNN: Thermodynamic-Guided Graph Anomaly Detection with Synthetic Injection Framework for Imbalanced Node Classification
GANT-GNN is a novel thermodynamic-guided GNN for anomaly detection in imbalanced node classification. It combines physics-inspired principles with graph learning and uses a synthetic injection framework to handle class imbalance, achieving superior anomaly detection performance.

# Core dependencies
torch>=1.12.0
torch-geometric>=2.0.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0

# Graph processing
networkx>=2.6
ogb>=1.3.0

# Optimization and utilities
tqdm>=4.62.0
tensorboard>=2.7.0


# Clone the repository
git clone https://github.com/afofanah/gant-gnn.git
cd gant-gnn
pip install -r requirements.txt
conda create -n gantgnn python=3.8
conda activate gantgnn

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
pip install -r requirements.txt

#Command Line Interface
# Single experiment
python run_exp.py --mode single --dataset cora --use-physics

# Statistical analysis (5 runs)
python run_exp.py --mode statistical --datasets cora citeseer --num-runs 5

# Hyperparameter optimization
python run_exp.py --mode hyperopt --dataset cora --trials 20

# Physics vs baseline comparison
python run_exp.py --mode compare --datasets cora citeseer ogbn-arxiv --num-runs 5

# Complete evaluation on all datasets
python run_exp.py --mode all-datasets --use-physics

If you use this work in your research, please cite:
@article{fofanahgant,
  title={GANT-GNN: Thermodynamic-Guided Graph Anomaly Detection with Synthetic Feature Injection for Imbalanced Node Classification},
  author={Fofanah, Abdul Joseph and Wen, Lian and Chen, David and Yao, Tsungcheng and Sankoh, Albert Patrick and others},
  journal={Authorea Preprints},
  publisher={Authorea}
}
