# MAGE

To Create an environment:
```bash
conda env create -f environment.yml
```

To Reproduce the results of paper:
Please run 
```bash
python sample.py
```

To Run MAGE on personal dataset:

Run `python train_target_model.py` to create a new pretrained GNN.

Run `python main.py` to generate explanations of the target GNN.