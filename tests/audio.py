import torch
from amspp.modals import Image, Sensor
from amspp.mmdataset import schema, MultiModalDataset
# from torch.utils.data import Dataset

image = Image("tests/1.jpg", time_steps=6)
a = [0.33, 0.34, 0.34, 0.35, 0.37, 0.32]
sensor = Sensor(a, time_steps=6)

dataset = MultiModalDataset()
dataset.register_dataset("Eye", [image])
dataset.register_dataset("Accel", [sensor])

@schema
def collate(batch):
    image = batch["Eye"]
    accel = batch["Accel"]
    return torch.cat([image, accel], dim=1)

dataset.set_schema(collate)
