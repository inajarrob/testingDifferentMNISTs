import torch
import torch.nn as nn
import torch.optim as optim
from spikingjelly.clock_driven import encoding
from torchvision import transforms
from PIL import Image, ImageOps
import sys


def testModel():
    print ("Args %s" % (sys.argv[2]))  
    try:
        net = torch.load(sys.argv[1])
        print('\nloading pre-trained model...')
    except: #file doesn't exist yet
          print("na de na")
          pass

    net.eval()
    with torch.no_grad():
        print("\nloading the image...")
        img = Image.open(sys.argv[2])
        print("\nImg converted to gray")
        gray = ImageOps.grayscale(img)
        gray = gray.resize((28,28))
        gray.show()
        img = gray   
        convert_tensor = transforms.ToTensor()

        print("\nImg converted to tensor")
        T = 100
        encoder = encoding.PoissonEncoder()
        print("\nBefore the loop")
        for t in range(T):
            if t == 0:
                out_spikes_counter = net(encoder(convert_tensor(img).to("cuda:0")).float())
            else:
                out_spikes_counter += net(encoder(convert_tensor(img).to("cuda:0")).float())
        out_spikes_counter_frequency = (out_spikes_counter / T).cpu().numpy()
        print(f'Firing rate: {out_spikes_counter_frequency[0]}')

def main():
    testModel()

main()
