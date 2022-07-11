# testingDifferentMNISTs
This is a experiment that execute different neural networks to check their final results. This codes are implemented to compare ANN, CNN and SNN training times.

## Results CPU
- To execute this code you can find in ANN and CNN folders notebooks to execute for 10, 50 and 100 epochs.
- To execute the SNN you need to get the library [SpikingJelly](https://github.com/fangwei123456/spikingjelly) in the root SNN and install the requirements. To run the code use: `python lif_fc_mnist.py` and to change the epochs you can add to the sentence `--epoch=10` and `--device=cpu` to execute in CPU. It has been tested for 10, 50 and 100 epochs.

![image](https://user-images.githubusercontent.com/36012044/178204095-ae91407d-1b35-4b2c-9fe7-dd23fb790c5f.png)

## Results GPU (NVIDIA 1660 Ti Max)
- ANN: To execute this code you can find `python cnnMnistGPU.py numEpochs` in ANN folder and to run you need to pass how many epochs you want (10, 50 or 100 for example).
- CNN: To execute this code you can find `python GPUmodel.py numEpochs` in CNN folder and to run you need to pass how many epochs you want (10, 50 or 100 for example).
- SNN: You need to run: `python lif_fc_mnist.py` and to change the epochs you can add to the sentence you can add to the sentence `--epoch=10` and `--device=cuda:0` to execute in GPU. It has been tested for 10, 50 and 100 epochs.

![image](https://user-images.githubusercontent.com/36012044/178203989-1e57ca4a-4737-499a-b275-6966a7a3d747.png)

## Results changing batch size parameter
Another test you can run is changing the batch size. It can be modified in the code in ANN and CNN models in the code of the GPU.
In SNN you can add to the sentence `--batch-size=32`.

![image](https://user-images.githubusercontent.com/36012044/178258216-887711c1-30f1-4f8a-b3ab-3628b3a9067e.png)

## Testing
You can find in the folders different freeze models in notebooks to test the code obtained and in the folder examples some numbers to pass to the model.
