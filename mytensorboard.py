from torch.utils.tensorboard import SummaryWriter

# from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("log")

path = r"train/ants/5650366_e22b7e1065.jpg"
img = Image.open(path)
img_arr = np.array(img)

writer.add_image("hh1h", img_arr, 1, dataformats='HWC')
for i in range(100):
    writer.add_scalar("y=x", i, i)
    writer.add_scalar("y=x3", i, i ^ 3)
writer.close()
print("zasdfrnhjsaznt")

#       tensorboard --logdir=log --port=6007
