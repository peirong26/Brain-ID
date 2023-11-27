# Brain-ID: Learning Robust Feature Representations for Brain Imaging


<p align="center">
  <img src="./assets/showcase.png" alt="drawing", width="850"/>
</p>


## Environment
Training and evaluation environment: Python 3.11.4, PyTorch 2.0.1, CUDA 12.2. Run the following command to install required packages.
```
pip install -r requirements.txt
```

## Demo: Playing with Brain-ID Sythetic Generator
<p align="center">
  <img src="./assets/data_gen.png" alt="drawing", width="850"/>
</p>
```
python scripts/demo_synth.py # config file in ~/BrainID/cfgs/test/demo_synth.yaml
```


## Training on Synthetic Data
Use the following code to train a feature representation model on synthetic data: 
```
python scripts/train.py anat.yaml
```
We also support Slurm submission:
```
sbatch scripts/train.sh
```

## Evaluating on Real Data
Use the following code to fine-tune a task-specific model on real data, using pre-trained Brain-ID weights: 
```
python scripts/eval.py supv_seg.yaml
```
We also support Slurm submission:
```
sbatch scripts/eval.sh
```

## Download 
Brain-ID pre-trained model: [Google Drive](https://drive.google.com/file/d/1Hoy243gQIWrlIuYULtd2eryk4os-cLLZ/view?usp=sharing)

ADNI dataset ( cases): [Official website]()

Segmentation labels of ADNI dataset for synthetic data simulation: [Google Drive]()

## Citation
```bibtex
@InProceedings{Liu_2023_BrainID,
    author    = {Liu, Peirong and Puonti, Oula and Hu, Xiaoling and Alexander, Daniel C. and Iglesias, Juan E.},
    title     = {Brain-ID: Learning Robust Feature Representations for Brain Imaging},
    journal   = {arXiv},
    year      = {2023},
    volume    = {abs/},
}