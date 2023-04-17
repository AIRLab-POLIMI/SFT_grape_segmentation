# SFT_grape_segmentation
Code for experiments with Surgical Fine-tuning (SFT) for grape bunch segmentation.
The strategy of selectively tuning specific network layers was termed surgical fine-tuning in

```BibTeX
@inproceedings{lee2022surgical,
  title={Surgical Fine-Tuning Improves Adaptation to Distribution Shifts},
  author={Lee, Yoonho and Chen, Annie S and Tajwar, Fahim and Kumar, Ananya and Yao, Huaxiu and Liang, Percy and Finn, Chelsea},
  booktitle={NeurIPS 2022 Workshop on Distribution Shifts: Connecting Methods and Applications}
}
```

and is here extended to the case of instance segmentation through Mask R-CNN models.


## Docker 
We provide a dockerfile under `./SFT-docker` that extends the [Deepo Docker image](https://hub.docker.com/r/ufoym/deepo/):

```
#build from dockerfile 
docker build sft-docker /path/to/Dockerfile
docker run -t sft-docker # approx 8GB of GPU memory needed
```

We use the [Neptune API](https://neptune.ai/) for tracking model training. 
Be sure to create a Neptune project and to export the Neptune API token before running.



## Dependencies 

If installing from source, please consider the following dependencies: 

* Our custom detectron2 fork to apply SFT to MaskRCNN. 
Install steps so that custom version is imported at each call of the detectron2 library:
    ```
    git clone https://github.com/AIRLab-POLIMI/detectron2_grape_DA.git
    cd detectron2_grape_DA.git
    python -m pip install -e .
    PYTHONPATH=PYTHONPATH=/usr/lib/python3.8/site-packages/:/detectron2_grape_DA
    ```

* Module dependencies are listed in the `requirements.txt` file

* We use the [Neptune API](https://neptune.ai/) for tracking model training. 
   Creating a Neptune project and exporting the Neptune API token is a requirement for running this code.


## Datasets 
* The Embrapa Wine Grape Instance Segmentation Dataset (WGISD),
accessible at [https://github.com/thsant/wgisd](https://github.com/thsant/wgisd)

* The VINEyard Piacenza Image Collections (VINEPICs), which is the result of a collaboration between AIRLab (Politecnico di Milano) and Università Cattolica del Sacro Cuore archives robot-collected images from real vineyards and is publicly available at:
[Zenodo](zenodo-link)

## Getting started

The different fine-tuning configurations are provided under the `configs` folder.

Model checkpoints are accessible on [Google Drive](https://drive.google.com/drive/folders/17wql4DseY2JA6NUpLLX5dzKAWiSGC11I?usp=sharing)

For example, to surgically fine-tune only the second ResNet block on VINEPICs21 run: 

```
python train.py --dataset vinepics21 --mode SFT_res2 \
                --trainval_path /path/to/vinepics21/ --model_cfg ./configs/SFT_res2_FPN.yaml \
                --weights path/to/wgisd/weights
```

By default, this will create a folder named `vinepics21_SFT_res2` under `./data/results` that contains the best model checkpoints computed on the validation set.
Loss and AP plots are instead logged through the [Neptune API](https://neptune.ai/). 
Then to test on VINEPICs22R, for example, run:

```
python test.py --dataset vinepics22 --mode vinepics22R \
                --basepath /path/to/vinepics22 --test_path /path/to/vinepics22R \
                --model_cfg ./configs/SFT_res2_FPN.yaml --out_dir ./data/results/vinepics21_SFT_res2
                
```

Evaluation results are saved in the folder passed as output (e.g., `./data/results/vinepics21_SFT_res2` in this case).
