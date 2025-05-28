# Visibility Graph Learning Molde (VGL)

## Prepare Code Enviroment
1. Clone this git repository and change the directory to this repository:
```angular2html
git clone https://github.com/16061025/VGL.git
cd VGL
```
2. Install the required conda environment
```angular2html
conda env create -f VGLenvironment.yml
conda activate VGL
```

## Prepare DataSet
Download EEG data into your data directory.
```angular2html
YOUR/DATA/DIR
|
|- Alzheimer
|- bonn
   |- AET.mat
   |- train_label_Bonn.mat
|- brainlat
   |
   |- EEG data
      |
      |- 1_AD
      |- 2_bvFTD
   ...

```

Dataset preprocessing

|           | sample         | label |
|-----------|----------------|-------|
| Alzheimer | 256 node Graph | [0,1] |
| autsim    | 256 node Graph | [0,1]  |
| bonn      | 256 node Graph | [0,1]  |
| brainlat  |                | [0,1]  |
| DREAMER   | 256 node Graph | [0,1]  |
| Epilepsy  | 256 node Graph | [0,1]  |
| MDD       | 256 node Graph | [0,1]  |


## Run Model
Before running the bash commend, modify `--data_dir` to your dataset directory
```angular2html
python main.py --task lp --dataset disease_lp --model HGCN --lr 0.01 --dim 16 --num-layers 2 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 0 --manifold PoincareBall --normalize-feats 0 --log-freq 5 --data_dir "PATH/TO/DATA/DIR"
```
```angular2html
optional arguments:
  -h, --help            show this help message and exit
  --lr LR               learning rate
  --dropout DROPOUT     dropout probability
  --cuda CUDA           which cuda device to use (-1 for cpu training)
  --epochs EPOCHS       maximum number of epochs to train for
  --weight-decay WEIGHT_DECAY
                        l2 regularization strength
  --optimizer OPTIMIZER
                        which optimizer to use, can be any of [Adam, RiemannianAdam]
  --momentum MOMENTUM   momentum in optimizer
  --patience PATIENCE   patience for early stopping
  --seed SEED           seed for training
  --log-freq LOG_FREQ   how often to compute print train/val metrics (in epochs)
  --eval-freq EVAL_FREQ
                        how often to compute val metrics (in epochs)
  --save SAVE           1 to save model and logs and 0 otherwise
  --save-dir SAVE_DIR   path to save training logs and model weights (defaults to logs/task/date/run/)
  --sweep-c SWEEP_C
  --lr-reduce-freq LR_REDUCE_FREQ
                        reduce lr every lr-reduce-freq or None to keep lr constant
  --gamma GAMMA         gamma for lr scheduler
  --print-epoch PRINT_EPOCH
  --grad-clip GRAD_CLIP
                        max norm for gradient clipping, or None for no gradient clipping
  --min-epochs MIN_EPOCHS
                        do not early stop before min-epochs
  --task TASK           which tasks to train on, can be any of [lp, nc]
  --model MODEL         which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HyperGCN]
  --dim DIM             embedding dimension
  --manifold MANIFOLD   which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]
  --c C                 hyperbolic radius, set to None for trainable curvature
  --r R                 fermi-dirac decoder parameter for lp
  --t T                 fermi-dirac decoder parameter for lp
  --pretrained-embeddings PRETRAINED_EMBEDDINGS
                        path to pretrained embeddings (.npy file) for Shallow node classification
  --pos-weight POS_WEIGHT
                        whether to upweight positive class in node classification tasks
  --num-layers NUM_LAYERS
                        number of hidden layers in encoder
  --bias BIAS           whether to use bias (1) or not (0)
  --act ACT             which activation function to use (or None for no activation)
  --n-heads N_HEADS     number of attention heads for graph attention networks, must be a divisor dim
  --alpha ALPHA         alpha for leakyrelu in graph attention networks
  --double-precision DOUBLE_PRECISION
                        whether to use double precision
  --use-att USE_ATT     whether to use hyperbolic attention or not
  --local-agg LOCAL_AGG
                        whether to local tangent space aggregation or not
  --dataset DATASET     which dataset to use
  --val-prop VAL_PROP   proportion of validation edges for link prediction
  --test-prop TEST_PROP
                        proportion of test edges for link prediction
  --use-feats USE_FEATS
                        whether to use node features or not
  --normalize-feats NORMALIZE_FEATS
                        whether to normalize input node features
  --normalize-adj NORMALIZE_ADJ
                        whether to row-normalize the adjacency matrix
  --split-seed SPLIT_SEED
                        seed for data splits (train/test/val)
  --n_channels N_CHANNELS
                        number of EEG data channels
  --n_sections N_SECTIONS
                        split number of a channel
  --n_classes N_CLASSES
                        number of label classes
  --data_dir DATA_DIR   data path
  --VGL_lr VGL_LR       learning rate
  --VGL_epochs VGL_EPOCHS
                        maximum number of epochs to train for
  --VGL_seed VGL_SEED   seed for training
  --VGL_eval-freq VGL_EVAL_FREQ
                        how often to compute val metrics (in epochs)
  --VGL_save VGL_SAVE   1 to save model and logs and 0 otherwise
  --VGL_save_dir VGL_SAVE_DIR
                        path to save training logs and model weights (defaults to logs/task/date/run/)
  --VGL_batch_size VGL_BATCH_SIZE
                        batch size
  --device DEVICE       training device
  --mocha_feat_dim MOCHA_FEAT_DIM
                        mocha graph feat dim
  --mocha_n_nodes MOCHA_N_NODES
                        mocha graph n nodes
```