# SFT_grape_segmentation
Code for experiments with Surgical Fine-tuning (SFT) on the grape instance segmentation tasks.


##Dependencies 
Custom detectron2 fork to apply SFT to MaskRCNN 

`git clone https://github.com/AIRLab-POLIMI/detectron2_grape_DA.git`


##Google Colab
Notebook version available at [this link](https://colab.research.google.com/drive/1Vq_h7Wj76pGyuKoeePinuykTyyAlKqtH).

##Docker 
Single cmds for now, TODO: replace with custom image


##Datasets 
Current trials on the Embrapa Wine Grape Instance Segmentation Dataset (WGISD),
accessible at [https://github.com/thsant/wgisd](https://github.com/thsant/wgisd)

Under `./data`:

- `wgisd_original_split.zip` Original split with grape bunch/no grape bunch (binary) annotation format
- `wgisd_split_byvariety.zip` further split by grape variety, with 20% held out from training split for validation