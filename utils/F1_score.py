import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from model.model import ConvNet
from dataset.CT_DataSet import MyCTDataSet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64
num_classes = 2
dropout_prob = 0

# load model
model = ConvNet(num_classes,dropout_prob)
checkpoint = torch.load("../checkpoint/ConvNet_11.pth")
model.load_state_dict(checkpoint)

# load data
ds_val = MyCTDataSet("../dataset/val_images.npy", "../dataset/val_labels.npy")
val_loader = DataLoader(ds_val,batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

# eval
model.eval()
total_f1 = 0
correct_val = 0
for iteration,(image,label) in enumerate(val_loader):
    image = image.to(torch.float)
    label = label.squeeze(-1)

    output = model(image)
    pred = torch.argmax(output,dim=1)
    for i in range(batch_size):
        if pred[i] == label[i]:
            correct_val += 1
    pred = pred.detach()
    f1 = f1_score(label,pred)
    total_f1 += f1

average_f1 = total_f1/len(val_loader)
accuracy_val = 100 * correct_val / len(ds_val)
print(average_f1,accuracy_val)