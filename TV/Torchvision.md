# Object detection and segmentation using Torchvision

## Setup
Load modules on a Linux machine.
```{bash}
ml Python/3.9.6-GCCcore-11.2.0
ml IPython/7.26.0-GCCcore-11.2.0
ml cuDNN/8.9.7.29-CUDA-12.3.0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 # for GPU computation reproducibility
```
The last line is needed because of a RuntimeError: Deterministic behavior was enabled, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

Load a tv013 python virtual environment. 




## Stage 2: classification


```{bash}
export HSV_type=1
export diagnostic_type="Final"

# pick a GPU
export gpuid=2


# prediction, only needs to run on 1 GPU
for modelMod in _nopretrained _pretrained
do
    for cropVersion in SEG_sS1_strips_v4 SEG_sS1_strips_v6 DET_dS_strips
    do
        for test in validation test alltest
        do
      
          # Set the test_img_dir environment variable
          export test_img_dir=Image/${test}_${cropVersion}
        
          # Run the Python prediction script
          python TV/tv_CLS_predict.py --HSV $HSV_type --diagnostic_type $diagnostic_type --test_img_dir $test_img_dir \
            --the_model \
saved_tv13_models/CLS_HSV${HSV_type}_Final_2classes_${cropVersion}${modelMod}_seed0.pth,\
saved_tv13_models/CLS_HSV${HSV_type}_Final_2classes_${cropVersion}${modelMod}_seed1.pth,\
saved_tv13_models/CLS_HSV${HSV_type}_Final_2classes_${cropVersion}${modelMod}_seed2.pth \
            --save_to HSV${HSV_type}_${test}_${cropVersion}${modelMod}.csv

        done
    done
done


# ensemble predictions from 3 models
python TV/tv_CLS_en.py --HSV $HSV_type



```
