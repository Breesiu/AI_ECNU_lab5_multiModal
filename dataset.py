import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import ViltProcessor, ViltModel

class CustomDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image = []
        self.text = []
        self.label = []
        self._load_data(label_file)
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm", do_pad=True)
        # print(self.processor)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
    

        # # Apply transformations if provided
        # if self.transform is not None:
        #     image = self.transform(image)
        # print(self.image[index])
        # return self.processor(self.image[index], self.text[index], return_tensors="pt", padding='max_length', max_length=100), \
        #                 torch.tensor(self.label[index])
        return self.image[index], self.text[index], self.label[index]

    def _load_data(self, label_file):
        label_map = {'positive': 0, 'neutral': 1, 'negative': 2, 'null': -1}
        with open(label_file, 'r') as f:        
            next(f)
            for line in f:
                guid, label = line.strip().split(',')
                image_path = os.path.join(self.data_dir, f"{guid}.jpg")
                text_path = os.path.join(self.data_dir, f"{guid}.txt")
                # self.image.append(self._load_image(image_path))
                self.image.append(image_path)
                self.text.append(self._load_text(text_path))
                self.label.append(label_map[label])

    def _load_image(self, image_path):
        # Implement image loading logic based on your requirements
        # This is just a placeholder
        return Image.open(image_path)

    def _load_text(self, text_path):
        # Implement text loading logic based on your requirements
        # This is just a placeholder
        with open(text_path, 'r', encoding='latin-1') as file:
            content = file.read()
        return content

# Usage example
# data_dir = './data'  # Specify the path to the data directory
# label_file = './train.txt'  # Specify the path to the train.txt file
# dataset = CustomDataset(data_dir, label_file)

# # Accessing a single sample
# image, text, label = dataset[0]
# image = Image.open(image)
# print(image)

# print(dataset[0:4])

# for k, v in dataset[:4].items():
#     print(v.shape)