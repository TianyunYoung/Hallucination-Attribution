from torch.utils.data import Dataset
import os
import PIL

class NoCapsDataset(Dataset):
    def __init__(self, dataset_dir, domain='out-domain'):
        caption_path = '/data/ailin/nocaps/nocaps_val_4500_captions.json'
        
        
        self.caption_path = caption_path
        self.dataset_dir = dataset_dir
        self.domain = domain
        
        self.image_ids = self.load_image_ids()
        
        self.dataset = self.load_dataset()
        self.image_save_dir = self.dataset.first().filepath.rsplit('/', 1)[0]

        self.all_classes = self.dataset.default_classes
    
    def load_image_ids(self):
        import json
        image_ids = []
        with open(self.caption_path, 'r') as f:
            nocaps_info = json.load(f)
            nocaps_info_detail = nocaps_info['images']
            for item in nocaps_info_detail:
                if item['domain'] == self.domain:
                    image_ids.append(item['open_images_id'])
        return image_ids
        
    def load_dataset(self):
        import fiftyone as fo
        
        if self.domain == 'out-domain':
            ds = '{}/out_domain'.format(self.dataset_dir)
        elif self.domain == 'near-domain':
            ds = '{}/near_domain'.format(self.dataset_dir)
        
        
        # we only load validation set
        dataset = fo.zoo.load_zoo_dataset(
                      "open-images-v7",
                      split="validation",
                      image_ids = self.image_ids,
                      dataset_dir=ds,
                    #   overwrite=True,
                    #   download_if_needed=True,
                      dataset_name='nocaps-{}'.format(self.domain),
                  )

        # sample = dataset.first()
        # sample.filepath = ds + '/validation/data/' + sample.filepath.rsplit('/', 1)[-1]
        # sample.save()
        # for sample in dataset:
        #     sample.filepath = ds + '/validation/data/' + sample.filepath.rsplit('/', 1)[-1]
        #     sample.save()
        
        return dataset
    
    def get_image_id(self, index):
        return self.image_ids[index]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_save_dir, '{}.jpg'.format(image_id))
        sample = self.dataset.match({
            "filepath": image_path
        }).first()
        
        # print(image_path, sample.filepath)
        
        img = PIL.Image.open(sample.filepath).convert('RGB')
        positive_labels = [classification.label for classification in sample.positive_labels.classifications]
        seg_labels = [detection.label for detection in sample.detections.detections]

        all_labels = positive_labels + seg_labels
        all_labels = list(set(all_labels))
        
        return img, all_labels
    
    def get_sample_by_id(self, image_id):
        return self.__getitem__(self.get_index_by_image_id(image_id))

    def get_index_by_image_id(self, image_id):
        return self.image_ids.index(image_id)
    
    def get_raw_item_by_id(self, image_id):
        image_path = os.path.join(self.image_save_dir, '{}.jpg'.format(image_id))
        sample = self.dataset.match({
            "filepath": image_path
        }).first()
        return sample
