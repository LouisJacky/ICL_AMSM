r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader
import albumentations as A
import albumentations.pytorch
from PIL import Image
import numpy as np
import cv2
from .pascal import DatasetPASCAL
# from RL_base.Painter.data.pairdataset import PairDataset
# from RL_base.Painter.data import pair_transforms
# from RL_base.Painter.util.masking_generator import MaskingGenerator

from RL_base.open_flamingo.eval.eval_datasets import VQADataset

import torch

class Compose(A.Compose):
    def __init__(self, transforms, bbox_params=None, keypoint_params=None, additional_targets=None, p=1):
        super().__init__(transforms, bbox_params=bbox_params, keypoint_params=keypoint_params, additional_targets=additional_targets, p=p)

    def __call__(self, image, mask):
        augmented = super().__call__(image=np.array(image), mask=np.array(mask))
        return augmented['image'], augmented['mask']

class Compose_ofv2(A.Compose):
    def __init__(self, transforms, bbox_params=None, keypoint_params=None, additional_targets=None, p=1):
        super().__init__(transforms, bbox_params=bbox_params, keypoint_params=keypoint_params, additional_targets=additional_targets, p=p)

    def __call__(self, image):
        augmented = super().__call__(image=np.array(image))
        return augmented['image']

class FSSDataset:

    @classmethod
    def initialize(cls, benchmark, img_size, datapath, use_original_imgsize, apply_cats_augmentation=False, apply_pfenet_augmentation=False, args=None):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            # 'pair': PairDataset,
            "vizwiz": VQADataset
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        cats_augmentation = [
            A.ToGray(p=0.2),
            A.Posterize(p=0.2),
            A.Equalize(p=0.2),
            A.Sharpen(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.Solarize(p=0.2),
            A.ColorJitter(p=0.2),
        ]

        scale_limit = (0.9, 1.1) if benchmark == 'coco' else (0.8, 1.25)

        pfenet_augmentation = [
            A.RandomScale(scale_limit=scale_limit, p=1.),
            A.Rotate(limit=10, p=1.),
            A.GaussianBlur((5, 5), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT,
                value=[x * 255 for x in cls.img_mean], mask_value=0),
            A.RandomCrop(img_size, img_size),
        ]

        cls.trn_transform = Compose([
            *(cats_augmentation if apply_cats_augmentation else ()),
            *(pfenet_augmentation if apply_pfenet_augmentation else ()),
            A.Resize(img_size, img_size),
            A.Normalize(cls.img_mean, cls.img_std),
            A.pytorch.transforms.ToTensorV2(),
        ])

        cls.transform = Compose([
            A.Resize(img_size, img_size),
            A.Normalize(cls.img_mean, cls.img_std),
            A.pytorch.transforms.ToTensorV2(),
        ])

        cls.trn_transform_ofv2 = Compose_ofv2([
            *(cats_augmentation if apply_cats_augmentation else ()),
            *(pfenet_augmentation if apply_pfenet_augmentation else ()),
            A.Resize(img_size, img_size),
            A.Normalize(cls.img_mean, cls.img_std),
            A.pytorch.transforms.ToTensorV2(),
        ])

        cls.transform_ofv2 = Compose_ofv2([
            A.Resize(img_size, img_size),
            A.Normalize(cls.img_mean, cls.img_std),
            A.pytorch.transforms.ToTensorV2(),
        ])

        # # simple augmentation
        # cls.transform_train = pair_transforms.Compose([
        #     pair_transforms.RandomResizedCrop(img_size, scale=(args.min_random_scale, 1.0), interpolation=3),
        #     # 3 is bicubic
        #     pair_transforms.RandomApply([
        #         pair_transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
        #     ], p=0.8),
        #     pair_transforms.RandomHorizontalFlip(),
        #     pair_transforms.ToTensor(),
        #     pair_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # cls.transform_train2 = pair_transforms.Compose([
        #     pair_transforms.RandomResizedCrop(img_size, scale=(0.9999, 1.0), interpolation=3),  # 3 is bicubic
        #     pair_transforms.ToTensor(),
        #     pair_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # cls.transform_train3 = pair_transforms.Compose([
        #     pair_transforms.RandomResizedCrop(img_size, scale=(0.9999, 1.0), interpolation=3),  # 3 is bicubic
        #     pair_transforms.ToTensor(),
        #     pair_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # cls.transform_train_seccrop = pair_transforms.Compose([
        #     pair_transforms.RandomResizedCrop(img_size, scale=(args.min_random_scale, 1.0), ratio=(0.3, 0.7),
        #                                       interpolation=3),  # 3 is bicubic
        # ])
        # cls.transform_val = pair_transforms.Compose([
        #     pair_transforms.RandomResizedCrop(img_size, scale=(0.9999, 1.0), interpolation=3),  # 3 is bicubic
        #     pair_transforms.ToTensor(),
        #     pair_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        #
        # cls.masked_position_generator = MaskingGenerator(
        #     args.window_size, num_masking_patches=args.num_mask_patches,
        #     max_num_patches=args.max_mask_patches_per_block,
        #     min_num_patches=args.min_mask_patches_per_block,
        # )
        cls.args = args

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        # if benchmark == 'pair':
        #     # if split == 'trn':
        #     #     return cls.build_Pair_dataloader(benchmark, bsz, nworker, split, cls.args.data_path, cls.args.json_path)
        #     # else:
        #     #     return cls.build_Pair_dataloader(benchmark, bsz, nworker, split, cls.args.data_path, cls.args.val_json_path)
        #     return cls.build_Pair_dataloader(benchmark, bsz, nworker, split, cls.args.data_path, cls.args.json_path)
        if benchmark == "vizwiz":
            return cls.build_vizwiz_dataloader(benchmark, bsz, nworker, split, cls.args)

        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0
        transform = cls.trn_transform if split == 'trn' else cls.transform

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker, drop_last=True)

        return dataloader



    @classmethod
    def build_vizwiz_dataloader(cls, benchmark, bsz, nworker, split, args):
        transform = cls.trn_transform_ofv2 if split == 'trn' else cls.transform_ofv2

        train_image_dir_path = args.vizwiz_train_image_dir_path
        train_questions_json_path = args.vizwiz_train_questions_json_path
        train_annotations_json_path = args.vizwiz_train_annotations_json_path
        test_image_dir_path = args.vizwiz_test_image_dir_path
        test_questions_json_path = args.vizwiz_test_questions_json_path
        test_annotations_json_path = args.vizwiz_test_annotations_json_path

        if split == 'trn':
            dataset = cls.datasets[benchmark](
                image_dir_path=train_image_dir_path,
                question_path=train_questions_json_path,
                annotations_path=train_annotations_json_path,
                is_train=True,
                dataset_name="vizwiz",
                transform=transform,
            )
        else:
            dataset = cls.datasets[benchmark](
                image_dir_path=test_image_dir_path,
                question_path=test_questions_json_path,
                annotations_path=test_annotations_json_path,
                is_train=False,
                dataset_name="vizwiz",
                transform=transform,
            )

        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        # dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker, drop_last=True, )
        def custom_collate_fn(batch):
            collated_batch = {}
            for key in batch[0].keys():
                collated_batch[key] = [item[key] for item in batch]
            return collated_batch

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=custom_collate_fn,
        )

        # shuffle = split == 'trn'
        # nworker = nworker if split == 'trn' else 0
        # dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker, drop_last=True, collate_fn=custom_collate_fn, )

        return dataloader

    @classmethod
    def build_Pair_dataloader(cls, benchmark, bsz, nworker, split, data_path, json_path):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0
        transform = cls.trn_transform if split == 'trn' else cls.transform

        dataset = cls.datasets[benchmark](data_path, json_path, transform=cls.transform_train, transform2=cls.transform_train2, transform3=cls.transform_train3,
                                         transform_seccrop=cls.transform_train_seccrop, masked_position_generator=cls.masked_position_generator,
                                         use_two_pairs=cls.args.use_two_pairs, half_mask_ratio=cls.args.half_mask_ratio)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker, drop_last=True)

        return dataloader

    # @classmethod
    # def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
    #     # Force randomness during training for diverse episode combinations
    #     # Freeze randomness during testing for reproducibility
    #     if benchmark == 'FloodNet' or benchmark == 'aeroscapes' or benchmark == 'ottawa':
    #         return cls.build_iSALD_dataloader(benchmark, bsz, nworker, fold, split, shot=shot)
    #     if benchmark == 'iSALD':
    #         return cls.build_iSALD_5i_dataloader(benchmark, bsz, nworker, fold, split, shot=shot, use_original_imgsize=cls.use_original_imgsize)
    #     shuffle = split == 'trn'
    #     nworker = nworker if split == 'trn' else 0
    #     transform = cls.trn_transform if split == 'trn' else cls.transform
    #
    #     dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=transform, split=split, shot=shot, use_original_imgsize=cls.use_original_imgsize)
    #     dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker, drop_last=True)
    #
    #     return dataloader
    #
    # @classmethod
    # def build_part_dataloader(cls, benchmark, bsz, nworker, fold, split, input_size, shot=1):
    #     # Force randomness during training for diverse episode combinations
    #     # Freeze randomness during testing for reproducibility
    #     shuffle = split == 'trn'
    #     nworker = nworker if split == 'trn' else 0
    #     if split == 'trn':
    #         split_add = 'train'
    #     else:
    #         split_add = 'val'
    #
    #     dataset = cls.datasets[benchmark](input_size=input_size, split=split_add, dataset_root=cls.datapath, pascal_class=fold)
    #     dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker, drop_last=True)
    #
    #     return dataloader
    #
    # @classmethod
    # def build_iSALD_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
    #     # Force randomness during training for diverse episode combinations
    #     # Freeze randomness during testing for reproducibility
    #     shuffle = split == 'trn'
    #     nworker = nworker if split == 'trn' else 0
    #     transform = cls.trn_transform if split == 'trn' else cls.transform
    #     if split == 'trn':
    #         split = 'train'
    #
    #     dataset = cls.datasets[benchmark](cls.datapath, quality='semantic', mode=split, transform=transform, shot=shot)
    #     dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker, drop_last=True)
    #
    #     return dataloader
    #
    # @classmethod
    # def build_iSALD_5i_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1, use_original_imgsize=None):
    #     # Force randomness during training for diverse episode combinations
    #     # Freeze randomness during testing for reproducibility
    #     shuffle = split == 'trn'
    #     nworker = nworker if split == 'trn' else 0
    #     transform = cls.trn_transform if split == 'trn' else cls.transform
    #     if split == 'trn':
    #         split = 'train'
    #
    #     dataset = cls.datasets[benchmark](datapath=cls.datapath, fold=fold, transform=transform,
    #                                       shot=shot, mode=split, use_original_imgsize=cls.use_original_imgsize)
    #     dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker, drop_last=True)
    #
    #     return dataloader