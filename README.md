# [Brain-ID: Learning Robust Feature Representations for Brain Imaging](http://arxiv.org/abs/2311.16914)

**Peirong Liu<sup>1</sup>, Oula Puonti<sup>1,2</sup>, Xiaoling Hu<sup>1</sup>, Daniel C. Alexander<sup>3</sup>, Juan Eugenio Iglesias<sup>1,3,4</sup>**

<sup>1</sup> Athinoula A. Martinos Center for Biomedical Imaging, Harvard Medical School and Massachusetts General Hospital, Boston, USA <br />
<sup>2</sup> Danish Research Centre for Magnetic Resonance, Centre for Functional and Diagnostic Imaging and Research, Copenhagen University Hospital - Amager and Hvidovre, Copenhagen, Denmark <br />
<sup>3</sup> Centre for Medical Image Computing, University College London, London, UK <br />
<sup>4</sup> Computer Science and Artificial Intelligence Laboratory, Massachusetts Institute of Technology, Cambridge, USA

<p align="center">
  <img src="./assets/showcase.png" alt="drawing", width="850"/>
</p>


## Environment
Training and evaluation environment: Python 3.11.4, PyTorch 2.0.1, CUDA 12.2. Run the following command to install required packages.
```
conda create -n brainid python=3.11
conda activate brainid

git clone https://github.com/peirong26/Brain-ID
cd /path/to/brain-id
pip install -r requirements.txt
```
Please also install [pytorch3dunet](https://github.com/wolny/pytorch-3dunet) according to the official instructions.

## Demo
### Playing with Brain-ID Sythetic Generator
<p align="center">
  <img src="./assets/data_gen.png" alt="drawing", width="850"/>
</p>

```
python scripts/demo_synth.py
```
You can customize your own data generator in `cfgs/demo_synth.yaml`.

### Playing with Brain-ID Feature Extractor
You can input your own images, and obtain their Brain-ID features using the following code.
```
import torch
from utils.demo_utils import prepare_image, get_feature

img_path = '/path/to/your/test/image' 
ckp_path = '/path/to/Brain-ID/pre-trained/weights'

im = prepare_image(img_path, device='cuda:0')
feats = get_feature(im, ckp_path, device='cuda:0')
```

## Training on Synthetic Data

<p align="center">
  <img src="./assets/framework.png" alt="drawing", width="850"/>
</p>

Use the following code to train a feature representation model on synthetic data: 
```
python scripts/train.py anat.yaml
```
We also support Slurm submission:
```
sbatch scripts/train.sh
```
You can customize your anatomy supervision by changing the configure file `cfgs/train/anat.yaml`. We provide two other anatomy supervision choices in `cfgs/train/seg.yaml` and `cfgs/train/anat_seg.yaml`.

## Evaluating on Real Data
Use the following code to fine-tune a task-specific model on real data, using Brain-ID pre-trained weights: 
```
python scripts/eval.py task_recon.yaml
```
We also support Slurm submission:
```
sbatch scripts/eval.sh
```
The argument `task_recon.yaml` configures the task (anatomy reconstruction) we are evaluating. We provide other task-specific configure files in `cfgs/train/task_seg.yaml` (brain segmentation), `cfgs/train/anat_sr.yaml` (image super-resolution), and `cfgs/train/anat_bf.yaml` (bias field estimation). You can customize your own task by creating your own `.yaml` file.

## Download 
- Brain-ID pre-trained model: [Google Drive](https://drive.google.com/file/d/1Hoy243gQIWrlIuYULtd2eryk4os-cLLZ/view?usp=sharing)

- ADNI, ADNI3 and AIBL datasets: Request data from [official website](https://adni.loni.usc.edu/data-samples/access-data/).

- ADHD200 dataset: Request data from [official website](https://fcon_1000.projects.nitrc.org/indi/adhd200/).

- HCP dataset: Request data from [official website](https://www.humanconnectome.org/study/hcp-young-adult/data-releases).

- OASIS3 dataset Request data from [official website](https://www.oasis-brains.org/#data).

- Segmentation labels for data simulation: To train a Brain-ID feature representation model of your own from scratch, one needs the segmentation labels for synthetic image simulation and their corresponding MP-RAGE images for anatomy supervision. We provide our processed segmentation labels for synthetic data simulation [here](https://drive.google.com/drive/folders/1PTr-gp7RD-4CzbzgRF0ZVLnAEFLAd29U?usp=sharing). The file names correspond to subject names in [ADNI](https://adni.loni.usc.edu/) dataset. The ground truth T1-weighted, MP-RAGE images need to be requested and downloaded from ADNI's official website as listed above.


## Datasets
After downloading the datasets needed, structure the data as follows, and set up your dataset paths in `BrainID/datasets/__init__.py`.
```
/path/to/dataset/
  T1/
    subject_name.nii
    ...
  T2/
    subject_name.nii
    ...
  FLAIR/
    subject_name.nii
    ...
  CT/
    subject_name.nii
    ...
  or_any_other_modality_you_have/
    subject_name.nii
    ...
  segmentation_maps/
    subject_name.nii
    ...
```

## Citation
```bibtex
@InProceedings{Liu_2023_BrainID,
    author    = {Liu, Peirong and Puonti, Oula and Hu, Xiaoling and Alexander, Daniel C. and Iglesias, Juan E.},
    title     = {Brain-ID: Learning Robust Feature Representations for Brain Imaging},
    journal   = {arXiv},
    year      = {2023},
    volume    = {abs/2311.16914},
}
