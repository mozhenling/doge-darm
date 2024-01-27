# DARM: Distance Aware Risk Minimization 

The repository contains the experiment implementations of the manuscript entitled
_"Distance Aware Risk Minimization for Domain Generalization in Machine Fault Diagnosis"_, 
which are currently under a peer review process.

## Run the codes
The repository is developed from the 
[DomainBed](https://github.com/facebookresearch/DomainBed). 
Therefore, running the codes is similar to that introduced in the DomainBed. An example of 
running the hyperparameter and model selections is given in `eg_sweep.sh`. After that, use 
`eg_results.sh` to collect the domain generalization results. See `requirements.txt` for 
recommended versions of dependent packages.


## Datasets
Datasets used in the paper are available as follows.

+ [Google Drive](https://drive.google.com/file/d/1ND1t5MFSQNlDg6xyEZsFstZCLkJ5CDwc/view?usp=sharing)  
+ or, [Baidu Netdisk](https://pan.baidu.com/s/1q94NaqTNacdrmjhuZqyu8g?pwd=y9qr)
& Password：y9qr

After downloading, place each data folder under _"./datasets"_.

## Acknowledgement 
We herein acknowledge the authors of the Domainbed for offering the code base.
We would also like to thank authors of each dataset for sharing the data.
The original data sources are listed as follows.

+ [CU_Actuator](https://cord.cranfield.ac.uk/articles/dataset/Data_set_for_Data-based_Detection_and_Diagnosis_of_Faults_in_Linear_Actuators_/5097649)
+ [CWRU_Bearing](https://engineering.case.edu/bearingdatacenter) 
+ [PHM_Gear](https://phmsociety.org/data-analysis-competition/) 
+ [UBFC_Motor](http://dx.doi.org/doi:10.25666/DATAUBFC-2023-03-06-03) 

