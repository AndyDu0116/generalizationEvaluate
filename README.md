# GeneralizationEvaluate
Code implementation and example data of "[Methodology for Evaluating the Generalization of ResNet](https://www.mdpi.com/2076-3417/14/9/3951)"
## Features
- IoU-based generalization evaluation method for CNN
- Other method
  - [Spectral Norm](https://proceedings.neurips.cc/paper/2017/hash/b22b257ad0519d4500539da3c8bcf4dd-Abstract.html)
  - [Nuclear Norm](https://proceedings.mlr.press/v202/deng23e.html)
  - [EI](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b3847cda0c8cc0cfcdacf462dc122214-Abstract-Conference.html)

## Installation
```
conda create -n generalizationEval python=3.8 -y
conda activate generalizationEval
pip install -r requirements.txt
```
## Usage
### IoU-based
```
cd IoU_based
python RFRM_IoUbased.py \
    --model-arch ResNet18 \
    --model-file ../dataExample/modelSet/ResNet18_example1.pth \
    --data-root ../dataExample/imageData \
    --cam gradcam # CAM method, choices=['gradcam', 'gradcampp','smoothgradcampp', 'layercam']
```
### Spectral Norm
```
cd SpectralNorm
python spectral_norm.py \
    --model-arch ResNet18 \
    --model-file ../dataExample/modelSet/ResNet18_example1.pth \
    --data-root ../dataExample/imageData
```
### Nuclear Norm
```
cd NuclearNorm
python nuclear_norm.py \
    --model-arch ResNet18 \
    --model-file ../dataExample/modelSet/ResNet18_example1.pth \
    --data-root ../dataExample/imageData
```
### EI
```
cd EI
python EI_score_rotation.py \
    --model-arch ResNet18 \
    --model-file ../dataExample/modelSet/ResNet18_example1.pth \
    --data-root ../dataExample/imageData
```
## Reference
```
@article{du2024methodology,
  title={Methodology for Evaluating the Generalization of ResNet},
  author={Du, Anan and Zhou, Qing and Dai, Yuqi},
  journal={Applied Sciences},
  volume={14},
  number={9},
  pages={3951},
  year={2024},
  publisher={MDPI}
}
```
