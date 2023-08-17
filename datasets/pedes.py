import torch.utils.data as data
import numpy as np
import os
import pickle
# import h5py
import json
from PIL import Image
from utils.directory import check_exists
from scipy.misc import imread, imresize
import datasets.preprocess as preprocess

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class CuhkPedes(data.Dataset):
    '''
    Args:
        root (string): Base root directory of dataset where [split].pkl and [split].h5 exists
        split (string): 'train', 'val' or 'test'
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed vector. E.g, ''transform.RandomCrop'
        target_transform (callable, optional): A funciton/transform that tkes in the
            targt and transfomrs it.
    '''

    def __init__(self, image_root, anno_root, split, max_length, transform=None, target_transform=None, \
                 cap_transform=None, vocab_path='', min_word_count=0):

        self.image_root = image_root
        self.anno_root = anno_root
        self.max_length = max_length
        self.transform = transform
        self.target_transform = target_transform
        self.cap_transform = cap_transform
        self.split = split.lower()
        self.vocab_path = vocab_path
        self.min_word_count = min_word_count

        if not check_exists(self.image_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               'Please follow the directions to generate datasets')

        data = self.get_data_from_json()
        self.read_data(data)

    
    def add_caption_to_data(self, split_data, data):

        fname_dict = {jj['file_path']:jj for jj in split_data}
        judge = {jj['file_path']: 0 for jj in split_data}

        for fname in data['images_path']:
            judge[fname] = judge[fname] + 1
        
        caption = []
        for fname in data['images_path']:
            jj = fname_dict[fname]

            caption.append(jj['captions'][judge[fname] - 1])
            judge[fname] = judge[fname] - 1
            
            
        assert len(caption) == len(data['images_path'])
        data['captions'] = caption
        return data

    
    def get_data_from_json(self):
        args = Namespace(min_word_count=self.min_word_count, remove_stopwords = None, out_root=None)

        split_data = self.load_split(self.split)
       
        if self.vocab_path == '':
            vocab = preprocess.build_vocab(split_data, args, write=False)
        else:
            vocab = self.load_vocab(self.vocab_path)

       
        split_metadata = preprocess.process_metadata(self.split, split_data, args, write=False)
        split_decodedata = preprocess.process_decodedata(split_metadata, vocab)
        data = preprocess.process_dataset(self.split, split_decodedata, args, write=False)
        

        data = self.add_caption_to_data(split_data, data)
        return data

    def load_split(self, split):   
        split_root = os.path.join(self.anno_root, split + '_reid.json')
        with open(split_root, 'r') as f:
            split_data = json.load(f)
        
        return split_data

    def load_vocab(self, vocab_path):    
        with open(os.path.join(vocab_path), 'rb') as f:
            word_to_idx = pickle.load(f)

        vocab = preprocess.Vocabulary(word_to_idx, len(word_to_idx))
        return vocab   

    def read_data(self, data):
        
        if self.split == 'train':
            self.train_labels = data['labels']
            self.train_captions = data['captions']         
            self.train_images = data['images_path']

        elif self.split == 'val':
            self.val_labels = data['labels']
            self.val_captions = data['captions']
            self.val_images = data['images_path']

        elif self.split == 'test':
            self.test_labels = data['labels']
            self.test_captions = data['captions']
            self.test_images = data['images_path']

            unique = []
            new_test_images = []
            for test_image in self.test_images:
                if test_image in new_test_images:
                    unique.append(0)
                else:
                    unique.append(1)
                    new_test_images.append(test_image)
            self.unique = unique

        
        else:
            raise RuntimeError('Wrong split which should be one of "train","val" or "test"')


    def __getitem__(self, index):
        """
        Args:
              index(int): Index
        Returns:
              tuple: (images, labels, captions)
        """
        if self.split == 'train':
            img_path, caption, label = self.train_images[index], self.train_captions[index], self.train_labels[index]
        elif self.split == 'val':
            img_path, caption, label = self.val_images[index], self.val_captions[index], self.val_labels[index]
        else:
            img_path, caption, label = self.test_images[index], self.test_captions[index], self.test_labels[index]


        img_path=os.path.join(self.image_root,img_path) 

        img = imread(img_path)
        if len(img.shape) == 2:
            img = np.dstack((img,img,img))
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, caption, label
    def __getitem__(self, index):
        """
        Args:
              index(int): Index
        Returns:
              tuple: (images, labels, captions)
        """
        if self.split == 'train':
            img_path, caption, label = self.train_images[index], self.train_captions[index], self.train_labels[index]
        elif self.split == 'val':
            img_path, caption, label = self.val_images[index], self.val_captions[index], self.val_labels[index]
        else:
            img_path, caption, label = self.test_images[index], self.test_captions[index], self.test_labels[index]


        img_path=os.path.join(self.image_root,img_path)

        img = imread(img_path)
        if len(img.shape) == 2:
            img = np.dstack((img,img,img))
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, caption, label
    def get_image(self, index):
        """
        Args:
              index(int): Index
        Returns:
              tuple: (images, labels, captions)
        """
        if self.split == 'train':
            img_path, caption, label = self.train_images[index], self.train_captions[index], self.train_labels[index]
        elif self.split == 'val':
            img_path, caption, label = self.val_images[index], self.val_captions[index], self.val_labels[index]
        else:
            img_path, caption, label = self.test_images[index], self.test_captions[index], self.test_labels[index]


        img_path=os.path.join(self.image_root,img_path)

        img = imread(img_path)
        if len(img.shape) == 2:
            img = np.dstack((img,img,img))
        img = Image.fromarray(img)

        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # if self.target_transform is not None:
        #     label = self.target_transform(label)

        return img, caption, label
    def __len__(self):
        if self.split == 'train':
            return len(self.train_labels)
        elif self.split == 'val':
            return len(self.val_labels)
        else:
            return len(self.test_labels)
