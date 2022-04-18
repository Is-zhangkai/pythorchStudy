from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter



img_path = r"train/ants/6240338_93729615ec.jpg"
writer = SummaryWriter("log")
img = Image.open(img_path)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
# writer.add_image("tensor",tensor_img,1)
#
# print(tensor_img[0][0][0])
# tn=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
# img_tn=tn(tensor_img)
# print(img_tn[0][0][0])
# writer.add_image("tensor1",img_tn,2)

# print(img.size)
# trans_resize=transforms.Resize(1024)
# img_resize=trans_resize(img)
# print(img_resize.size)
# img_resize=tensor_trans(img_resize)
# writer.add_image("resize",img_resize,2)


print(img.size)
trans_resize2=transforms.Resize(512)
trans_compose=transforms.Compose([trans_resize2,tensor_trans])
img_compose=trans_compose(img)

writer.add_image("resize",img_compose,3)



writer.close()
#       tensorboard --logdir=log --port=6007


