# Tensor Cores VS CUDA Cores Experiment 


This is an experiment to test deployment of hybrid Neural Network model (FC + CNN) inside a GPU using CUDA code and run that in Tensor Cores and CUDA cores
and compare the performance of simple inference engine over different GPUS. 


### Build 

To build 

```
cd inference
mkdir build && cd build
cmake .. && make -j4
```

if you need only to build for `fp16` or `fp32` then you need to do 
```
cmake -DBUILD_FP16_ONLY=ON .. && make
```

or 
```
cmake -DBUILD_FP32_ONLY=ON .. && make
```

**Note** I have conditional compilation depening on if the machine is one of our group cluster machine `sleepy` or `sneezy` in `CMakeLists.txt` because of the old and strange situation of these machines.
so to run on different machines you have to think about how to adjust `CMakeLists.txt`

### Running the inference 

Each of the binaries takes two arguments, the first is device id (GPU id number in case of multiple GPUs) and the second is `repeat factor` which define how many iteration of
CIFAR10 10000 images of validation data you want to run over (This is experiment about throughput not accuracy)

Example

```
./inference_cuda_fp16 0 1
```


There is a file in scripts folder called `run_inference.sh` which automate running this over our physics group machines `sleepy` and `sneezy` which have different GPUs
which you can adjust into your case. 

### Analysis 

for my particular experiment, I have created python script to do the analysis and produce the useful plots in `scripts/analysis.py`. 
