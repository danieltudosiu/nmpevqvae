# Neuromorphologicaly Preserving Volumetric Data Encoding Using VQ-VAE

This is a PyTorch port for the TensorFlow implementation of Neuromorphologicaly-preserving Volumetric data encoding using VQ-VAE.

I have not yet recreated the experiments from my paper. I am in the process of doing so for the 192x256x192 ones since they are of the most interest to the community. Once they are done I will also release pre-trained models as checkpoints for this repo.

## How to install  

I assume you are running a system Python version 3.6, otherwise please create a virtual environment with Python 3.6 

```bash
git clone https://github.com/danieltudosiu/nmpevqvae.git 
cd nmpevqvae
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt

cd ../
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Experiments
Overall command is:
```bash
python3 experiment.py --training_data_directory /path/to/healthy/training/dataset/ --testing_data_directory /path/to/healthy/testing/dataset/ --project_directory /path/to/project/output/directory/ --experiment_name Healthy_Adaptive --device 1 --mode Training --starting_iteration 0 --epochs 20000 -log_every 10000 --checkpoint_every 10000 --checkpoint_last 5 --batch_size 2 --learning_rate 0.0001 --loss Adaptive --reconstruction_lambda 1.0 --zero_image_gradient_loss 100000 --one_image_gradient_loss 10000 --max_image_gradient_loss 5 --first_decay_steps 6480 --alpha 0.0000001 --t_mul 1.25 --m_mul 0.95
```

For fine-tuning please copy the baseline experiment directory, rename it, and start from it by changing the ``--starting_iteration`` to ``-1`` and ``--project_directory`` to the newly renamed folder

The MMD and MS-SSIM calculations were done using the code provided by [1] which can be found [here](https://github.com/cyclomon/3dbraingen)

For the Voxel-Based Morphometry we have used SPM 12 [2] and the Dice calculation was done based on SPM segmentation using NiftySeg [3]. 

[1] Kwon, G., Han, C. and Kim, D.S., 2019, October. Generation of 3D brain MRI using auto-encoding generative adversarial networks. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 118-126). Springer, Cham.

[2] https://www.fil.ion.ucl.ac.uk/spm/software/spm12/

[3] Cardoso, M., Clarkson, M., Modat, M. and Ourselin, S., 2012. NiftySeg: open-source software for medical image segmentation, label fusion and cortical thickness estimation. In IEEE International Symposium on Biomedical Imaging, Barcelona, Spain.
## Citing
### Harvard

Tudosiu, P.D., Varsavsky, T., Shaw, R., Graham, M., Nachev, P., Ourselin, S., Sudre, C.H. and Cardoso, M.J., 2020. Neuromorphologicaly-preserving Volumetric data encoding using VQ-VAE. arXiv preprint arXiv:2002.05692.

### Bibtex

```
@article{tudosiu2020neuromorphologicaly,
  title={Neuromorphologicaly-preserving Volumetric data encoding using VQ-VAE},
  author={Tudosiu, Petru-Daniel and Varsavsky, Thomas and Shaw, Richard and Graham, Mark and Nachev, Parashkev and Ourselin, Sebastien and Sudre, Carole H and Cardoso, M Jorge},
  journal={arXiv preprint arXiv:2002.05692},
  year={2020}
}
```

## Contact

For any problems or questions in regards to this project please [open an issue](https://github.com/danieltudosiu/nmpevqvae/issues/new) in github.

For collaboration purposes please [e-mail me](mailto:petru.tudosiu@kcl.ac.uk).

## Licence

```
MIT License

Copyright (c) 2020 Petru-Daniel Tudosiu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
