FROM ufoym/deepo:latest

#Install custom detectron2 module
RUN git clone https://github.com/AIRLab-POLIMI/detectron2_grape_DA
WORKDIR ./detectron2_grape_DA
RUN python -m pip install -e .
RUN PYTHONPATH=PYTHONPATH=/usr/lib/python3.8/site-packages/:/detectron2_grape_DA

#Clone code for Surgical Fine-Tuning (SFT)
WORKDIR /
RUN git clone https://github.com/AIRLab-POLIMI/SFT_grape_segmentation.git
WORKDIR /SFT_grape_segmentation
#Install additional requirements not provided in Deepo image
RUN python -m pip install pyyaml==5.1
RUN pip install albumentations neptune split-folders




