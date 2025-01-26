import re
import os
import sys
sys.path.append("..")
import time
import h5py
import clip
import errno
import torch
import random
import numpy as np
import pandas as pd

import clip
from PIL import Image
from datetime import datetime
from alive_progress import alive_bar
from cluster_based_policy_feature import cluster_to_fix, cluster_nearest_farthest, cluster_nearest_farthest2

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)
Cyan = (0, 255, 255)
Color_map = {0: BLACK,
             1: WHITE,
             2: RED,
             3: GREEN,
             4: BLUE,
             5: YELLOW,
             6: PURPLE,
             7: Cyan}


def calculate_metric(target, ours, fg_color=WHITE, bg_color=BLACK):
    fg_color = np.array(fg_color)
    # Calculate accuracy:
    accuracy = np.sum(np.float32((target == ours).all(axis=2))) / (ours.shape[0] * ours.shape[1])
    seg_orig = ((target - fg_color[np.newaxis, np.newaxis, :]) == 0).all(axis=2)
    seg_our = ((ours - fg_color[np.newaxis, np.newaxis, :]) == 0).all(axis=2)
    iou = np.sum(np.float32(seg_orig & seg_our)) / np.sum(np.float32(seg_orig | seg_our))

    return {'iou': iou, 'accuracy': accuracy}


def prepare_model(model_attr, chkpt_dir, arch='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1', seg_type=None):
    # build model
    model = getattr(model_attr, arch)()
    if seg_type:
        model.seg_type = seg_type
    # model.to("cuda")
    # # load model
    # checkpoint = torch.load(chkpt_dir, 'cpu')
    checkpoint = torch.load(chkpt_dir)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model

def round_image(img, options=(WHITE, BLACK, RED, GREEN, BLUE), outputs=None, t=(0, 0, 0)):
    # img.shape == [224, 224, 3], img.dtype == torch.int32
    img = torch.tensor(img)
    t = torch.tensor((t)).to(img)
    options = torch.tensor(options)
    opts = options.view(len(options), 1, 1, 3).permute(1, 2, 3, 0).to(img)
    nn = (((img + t).unsqueeze(-1) - opts) ** 2).float().mean(dim=2)
    nn_indices = torch.argmin(nn, dim=-1)
    if outputs is None:
        outputs = options
    res_img = outputs.clone().detach()[nn_indices]
    # res_img = torch.tensor(outputs)[nn_indices]
    return res_img


# hook functions have to take these 3 input
def hook_forward_fn(module, input, output):
    print("It's forward: ")
    print(f"module: {module}")
    print(f"input: {input}")
    print(f"output: {output}")
    print("="*20)

def hook_backward_fn(module, grad_input, grad_output):
    print("It's backward: ")
    print(f"module: {module}")
    print(f"grad_input: {grad_input}")
    print(f"grad_output: {grad_output}")
    print("="*20)

def add_image_path(datapath, img_path, ann_path, query_name, chose_support_name, class_sample, args):
    chose_support_name = [os.path.basename(chose_name) for chose_name in chose_support_name]
    support_image = [(os.path.join(img_path, chose_id) + '.jpg') for chose_id in chose_support_name]
    support_cmask = [Image.open(os.path.join(ann_path, chose_id) + '.png') for chose_id in chose_support_name]
    query_image_name = os.path.join(img_path, query_name[0]) + '.jpg'
    query_cmask = Image.open(os.path.join(ann_path, query_name[0]) + '.png')
    real_class_id = class_sample.cpu().numpy()[0]

    query_mask, support_masks = get_query_support_mask(query_image_name, query_cmask,
                                                       support_image, support_cmask, real_class_id)

    return query_image_name, support_image, query_mask, support_masks

def add_image_path_painter(datapath, img_path, ann_path, query_name, chose_support_name, class_sample, args):
    chose_support_name = [os.path.basename(chose_name) for chose_name in chose_support_name]
    support_image = [(os.path.join(datapath+img_path, chose_id)) for chose_id in chose_support_name]
    support_cmask = [os.path.join(datapath+ann_path, chose_id) for chose_id in chose_support_name]
    query_image_name = os.path.join(datapath, query_name[0])
    query_cmask = os.path.join(datapath, query_name[0])
    real_class_id = class_sample.cpu().numpy()[0]

    query_mask, support_masks = query_cmask, support_cmask

    return query_image_name, support_image, query_mask, support_masks


def instance_num(query_mask):
    length = len(query_mask)
    a = [np.unique(query_mask[i], axis=0) for i in range(length)]
    b = a[0]
    for i in range(1, length):
        b = np.concatenate([b, a[i]])
    now = np.unique(b, axis=0)
    # index = np.where(now == [0, 0, 0])
    index = (now == 0).all(axis=1) == False
    # c = now[index]
    return len(now), now[index]

def change_map_color(query_ins_cmask, query_ins_cmaskr, class_real_id):
    query_ins_num, query_ins = instance_num(query_ins_cmaskr)
    query_ins_raw = np.array(query_ins_cmask)
    query_ins_raw = np.repeat(query_ins_raw.reshape(query_ins_raw.shape[0], query_ins_raw.shape[1], 1), 3, axis=-1)
    for i in range(1, query_ins_num):
        query_ins_cmaskr[(query_ins_raw == class_real_id[i]).all(-1)] = Color_map[i]
    return query_ins_cmaskr

def miou_evaluation(query_name, chose_support_name, query_mask, output_image_mask,
                    total_sample_num, start_time, miou, m_color_blind_iou, m_accuracy,
                    idx, logger, class_sample, args):
    if isinstance(query_mask, Image.Image):
        image_mask = query_mask.convert("RGB")
    else:
        image_mask = Image.open(query_mask).convert("RGB")

    # 2. Evaluate prediction
    image_mask = np.uint8(np.array(image_mask))
    image_mask = round_image(image_mask, [WHITE, BLACK])
    output_image_mask = round_image(output_image_mask, [WHITE, BLACK], t=args.t)
    end_time = time.time()
    batch_time = end_time - start_time
    current_metric = calculate_metric(image_mask, output_image_mask)
    logger.write(
        f"{datetime.now()} Batch: {idx}/{total_sample_num}, iou: {current_metric['iou']:.3f}, 'accuracy':{current_metric['accuracy']:.3f}, time_cost:{batch_time:.3f}")
    logger.write(f"Query image name: {query_name[0]}, class_id:{class_sample.cpu().numpy()[0]}")
    logger.write(f"Support image name:{chose_support_name}, Support_class_id:{class_sample.cpu().numpy()[0]}")

    miou += current_metric['iou']
    m_accuracy += current_metric['accuracy']
    return miou, m_accuracy


def new_extract_ignore_idx(image_name, masks, class_ids, purple=False):
    PURPLE = (0x44, 0x01, 0x54)
    YELLOW = (0xFD, 0xE7, 0x25)
    mask = np.array(masks)
    boundary = np.floor(mask / 255.)
    if not purple:
        if (class_ids + 1) not in mask:
            print(f'ohno, {image_name} not contain {class_ids}')
        mask[mask != class_ids + 1] = 0
        mask[mask == class_ids + 1] = 255
        return Image.fromarray(mask), boundary
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            if mask[x, y] != class_ids + 1:
                color_mask[x, y] = np.array(PURPLE)
            else:
                color_mask[x, y] = np.array(YELLOW)
    return Image.fromarray(color_mask), boundary


def get_query_support(query_name, class_sample, cand_prob, all_support_name, args):

    class_id = class_sample.clone().detach()
    class_id = class_id.cpu().numpy()
    # sample shot_pids from the cand_prob distribution
    cids = np.random.choice(range(len(cand_prob)), args.shot_number, p=cand_prob, replace=False)

    # img_path = os.path.join(args.base_dir, 'pascal-5i/VOC2012/JPEGImages/')
    # ann_path = os.path.join(args.base_dir, 'pascal-5i/VOC2012/SegmentationClassAug/')

    img_path = os.path.join(args.base_dir)
    ann_path = os.path.join(args.base_dir)

    choose_support_name = [all_support_name[cid] for cid in cids]

    # query_image = os.path.join(img_path, query_name) + '.jpg'
    # query_cmask = Image.open(os.path.join(ann_path, query_name) + '.png')

    query_image = os.path.join(img_path, query_name)
    query_cmask = Image.open(os.path.join(ann_path, query_name))
    query_cmask_path = os.path.join(ann_path, query_name)

    # support_image = [(os.path.join(img_path, chose_id) + '.jpg') for chose_id in choose_support_name]
    # support_cmask = [Image.open(os.path.join(ann_path, chose_id) + '.png') for chose_id in choose_support_name]

    support_image = [os.path.join(img_path, chose_id) for chose_id in choose_support_name]
    support_cmask = [Image.open(os.path.join(ann_path, chose_id)) for chose_id in choose_support_name]
    support_cmask_paths = [os.path.join(ann_path, chose_id) for chose_id in choose_support_name]

    real_class_id = class_id

    # query_mask, support_masks = get_query_support_mask(query_image, query_cmask, support_image, support_cmask, real_class_id)

    # return query_image, query_mask, support_image, support_masks, cids
    return query_image, query_cmask_path, support_image, support_cmask_paths, cids

def instance_num(query_mask):
    length = len(query_mask)
    a = [np.unique(query_mask[i], axis=0) for i in range(length)]
    b = a[0]
    for i in range(1, length):
        b = np.concatenate([b, a[i]])
    now = np.unique(b, axis=0)
    # index = np.where(now == [0, 0, 0])
    index = (now == 0).all(axis=1) == False
    # c = now[index]
    return len(now), now[index]

def change_map_color(query_ins_cmask, query_ins_cmaskr, class_real_id):
    query_ins_num, query_ins = instance_num(query_ins_cmaskr)
    query_ins_raw = np.array(query_ins_cmask)
    query_ins_raw = np.repeat(query_ins_raw.reshape(query_ins_raw.shape[0], query_ins_raw.shape[1], 1), 3, axis=-1)
    for i in range(1, query_ins_num):
        query_ins_cmaskr[(query_ins_raw == class_real_id[i]).all(-1)] = Color_map[i]
    return query_ins_cmaskr


def get_query_support_mask(query_image, query_cmask, support_image, support_cmask, class_id):
    query_mask, query_ignore_idx = new_extract_ignore_idx(query_image, query_cmask, class_id, purple=False)
    support_masks = []
    support_ignore_idxs = []
    for scmask_id in range(len(support_cmask)):
        support_mask, support_ignore_idx = new_extract_ignore_idx(support_image[scmask_id], support_cmask[scmask_id],
                                                                  class_id, purple=False)
        support_masks.append(support_mask)
        support_ignore_idxs.append(support_ignore_idx)
    return query_mask, support_masks

def get_query_support_mask_painter(query_image, query_cmask, support_image, support_cmask, class_id):
    query_mask, query_ignore_idx = new_extract_ignore_idx(query_image, query_cmask, class_id, purple=False)
    support_masks = []
    for scmask_id in range(len(support_cmask)):
        support_mask, support_ignore_idx = new_extract_ignore_idx(support_image[scmask_id], support_cmask[scmask_id],
                                                                  class_id, purple=False)
        support_masks.append(support_mask)
    return query_mask, support_masks

def get_query_support_contractive(query_name, class_sample, cand_prob, all_support_name, args):
    class_id = class_sample.clone().detach()
    class_id = class_id.cpu().numpy()
    # sample shot_pids from the cand_prob distribution
    cand_cids = torch.argsort(cand_prob, descending=True).cpu().detach().numpy().tolist()

    cand_positive_cids = cand_cids[:args.can_pos_num]
    cand_negative_cids = cand_cids[-args.can_neg_num:]

    positive_cids = random.sample(cand_positive_cids, args.shot_number)
    negative_cids = random.sample(cand_negative_cids, args.shot_number)

    cpos_suport_name = [all_support_name[cid] for cid in positive_cids]
    cneg_suport_name = [all_support_name[cid] for cid in negative_cids]

    img_path = os.path.join(args.base_dir, 'pascal-5i/VOC2012/JPEGImages/')
    ann_path = os.path.join(args.base_dir, 'pascal-5i/VOC2012/SegmentationClassAug/')

    query_image = os.path.join(img_path, query_name) + '.jpg'
    query_cmask = Image.open(os.path.join(ann_path, query_name) + '.png')

    cpos_support_image = [(os.path.join(img_path, chose_id) + '.jpg') for chose_id in cpos_suport_name]
    cneg_support_image = [(os.path.join(img_path, chose_id) + '.jpg') for chose_id in cneg_suport_name]
    cpos_support_cmask = [Image.open(os.path.join(ann_path, chose_id) + '.png') for chose_id in cpos_suport_name]
    cneg_support_cmask = [Image.open(os.path.join(ann_path, chose_id) + '.png') for chose_id in cneg_suport_name]

    real_class_id = class_id

    query_mask, cpos_support_masks = get_query_support_mask(query_image, query_cmask,
                                                            cpos_support_image, cpos_support_cmask, real_class_id)
    _, cneg_support_masks = get_query_support_mask(query_image, query_cmask,
                                                   cneg_support_image, cneg_support_cmask, real_class_id)

    support_images = cpos_support_image + cneg_support_image
    support_masks = cpos_support_masks + cneg_support_masks
    cids = positive_cids + negative_cids

    return query_image, query_mask, support_images, support_masks, cids


def process_to_retrieve_idx(class_sample, query_name, args):
    class_sample_id = class_sample.cpu().numpy()[0]
    query_select_name = os.path.basename(query_name[0])

    return query_select_name, class_sample_id



def retrieve_from_h5py(h5py_name, images, classes, args):
    with h5py.File(h5py_name, 'r') as f:
        all_boxes = []
        features = []
        images_name = []
        show_images_name = []  # cout element appear times
        class_images = images[classes]
        classes = str(classes)

        for image_path in class_images:
            image_name = os.path.basename(image_path)
            if not f[classes][image_name].__contains__(str(0)):
                features.append(f[classes][image_name]['features'][:])
                images_name.append(image_name)
            else:
                now_chose_id = show_images_name.count(image_name)
                features.append(f[classes][image_name][str(now_chose_id)]['features'][:])
                images_name.append(image_name)

            show_images_name.append(image_name)

    return features, images_name, all_boxes


# def save_h5py(VIT_name, features, class_name, sample_name, sample_path, sample_mask=None, sample_bbox=None):
#     dt = h5py.special_dtype(vlen=str)
#     with h5py.File(VIT_name, 'a') as f:
#         class_name = str(class_name)
#         if not f.__contains__(class_name):
#             subgroup = f.create_group(class_name)
#         else:
#             subgroup = f[class_name]
#
#         if not subgroup.__contains__(sample_name):
#             subsub = subgroup.create_group(sample_name)
#         else:
#             subsub = subgroup[sample_name]
#             if not subsub.__contains__(str(0)):
#                 path_0 = subsub['path'][:]
#                 features_0 = subsub['features'][:]
#                 subsub_0 = subsub.create_group(str(0))
#
#                 subsub_0.create_dataset('features', data=features_0)
#
#                 subsub_0.create_dataset('path', data=path_0)
#                 del subsub['features']
#
#
#
#             real_save_id = 1
#             while 1:
#                 if not subsub.__contains__(str(real_save_id)):
#                     subsub = subsub.create_group(str(real_save_id))
#                     break
#                 else:
#                     real_save_id = real_save_id + 1
#
#         subsub.create_dataset('features', data=features)
#         ds = subsub.create_dataset('path', sample_path.shape, dtype=dt)
#         ds[:] = sample_path

def save_h5py(VIT_name, features, class_name, sample_name, sample_path, question, sample_mask=None, sample_bbox=None):
    dt = h5py.special_dtype(vlen=str)
    with h5py.File(VIT_name, 'a') as f:
        class_name = str(class_name)
        if not f.__contains__(class_name):
            subgroup = f.create_group(class_name)
        else:
            subgroup = f[class_name]

        if not subgroup.__contains__(sample_name):
            subsub = subgroup.create_group(sample_name)
        else:
            subsub = subgroup[sample_name]
            if not subsub.__contains__(str(0)):
                path_0 = subsub['path'][:]
                features_0 = subsub['features'][:]
                # question_0 = subsub['question'][:]
                subsub_0 = subsub.create_group(str(0))

                subsub_0.create_dataset('features', data=features_0)
                subsub_0.create_dataset('path', data=path_0)
                # subsub_0.create_dataset('question', data=question_0)
                del subsub['features']
                # del subsub['question']

            real_save_id = 1
            while 1:
                if not subsub.__contains__(str(real_save_id)):
                    subsub = subsub.create_group(str(real_save_id))
                    break
                else:
                    real_save_id = real_save_id + 1

        subsub.create_dataset('features', data=features)
        ds = subsub.create_dataset('path', sample_path.shape, dtype=dt)
        ds[:] = sample_path
        # subsub.create_dataset('question', data=question)


def process_encoder_embedding(image, policy_model, confid_pos, device, args):
    embeddings = policy_model.preprocess_store(device, [image], [[image]],
                                               [[image]], args,
                                               is_process=True)  # len(input_list) x hidden_size
    embeddings = torch.cat((confid_pos, embeddings), dim=1)
    return embeddings



def retrieve_image(datapath, image_index, args):
    if '.jpg' not in image_index:
        image_index = image_index + '.jpg'
    image_path = os.path.join(datapath + '/VOC2012/JPEGImages', image_index)

    image = Image.open(image_path)

    return image, image_path

def retrieve_image_Painter(datapath, image_index, args):
    if '.jpg' not in image_index:
        # image_index = image_index["img_path"] + '.jpg'
        image_index = image_index["img_path"]
    image_path = os.path.join(datapath, image_index)

    image = Image.open(image_path)

    return image, image_path

def image_transform(image, transform, device, args):
    if (len(image.split()) != 3) or (image.mode != 'RGB'):
        print(image.mode)
        image = image.convert('RGB')
    image, _ = transform(image, image)
    # image = image.to(device)
    return image

def retrieve_raw_path(raw_path, args):
    return raw_path

def prepare_save_vit_h5py(Name, data, datapath, policy_model, transform, device, args, detection_model=None):
    if args.backbone == "clip":
        _, preprocess = clip.load(args.clip_ckpt_path, device=device)

    with alive_bar(len(data), title='SAVING···') as bar:
        for class_sample, choose_support_name in data.items():
            class_all_images = retrieve_raw_path(choose_support_name, args)

            for chose_id in range(len(class_all_images)):
                image, image_path = retrieve_image_Painter(datapath, class_all_images[chose_id], args)

                # If image is gray, transform into RGB image
                if args.backbone == "VIT":
                    image = image_transform(image, transform, device, args)

                # Except for Cand Sim method, directly generate feature embedding
                if args.backbone == "VIT":
                    embeddings = process_embedding(transform, image, device, policy_model)
                elif args.backbone == "clip":
                    embeddings = process_clip_embedding(preprocess, image, device, policy_model)

                features = embeddings.cpu().detach().numpy().astype(np.float32)

                sample_path = np.array([class_all_images[chose_id]["img_path"]])
                masks, bbox = None, None

                image_name = os.path.basename(class_all_images[chose_id]["img_path"])

                save_h5py(Name, features, class_sample, image_name, sample_path, masks, bbox)
            bar()

# def prepare_save_vit_h5py(Name, data, datapath, policy_model, transform, device, args, detection_model=None):
#     if args.backbone == "clip":
#         clip_model, preprocess = clip.load(args.clip_ckpt_path, device=device)
#
#     with alive_bar(len(data), title='SAVING···') as bar:
#         for class_sample, choose_support_name in data.items():
#             class_all_items = retrieve_raw_path(choose_support_name, args)
#
#             for item in class_all_items:
#                 image, image_path = retrieve_image_Painter(datapath, item['img_path'], args)
#                 question = item['question']
#
#                 # # 处理图像特征
#                 # if args.backbone == "VIT":
#                 #     image = image_transform(image, transform, device, args)
#                 #     image_embeddings = process_embedding(transform, image, device, policy_model)
#                 # elif args.backbone == "clip":
#                 #     image = preprocess(image).unsqueeze(0).to(device)
#                 #     image_embeddings = clip_model.encode_image(image).float()
#                 image_embeddings = process_image_embedding(image, device, clip_model, preprocess)
#
#
#                 # 处理文本特征
#                 # if args.backbone == "clip":
#                 #     text = clip.tokenize([question]).to(device)
#                 #     text_embeddings = clip_model.encode_text(text).float()
#                 # else:
#                 #     # 如果不是使用CLIP，您需要定义一个适合您模型的文本编码方法
#                 #     text_embeddings = process_text_embedding(question, device, policy_model)
#                 text_embeddings = process_text_embedding(question, device, clip_model, preprocess)
#
#                 # 合并图像和文本特征
#                 combined_features = torch.cat([image_embeddings, text_embeddings], dim=1)
#                 features = combined_features.cpu().detach().numpy().astype(np.float32)
#
#                 sample_path = np.array([item['img_path']])
#                 masks, bbox = None, None
#
#                 image_name = os.path.basename(item['img_path'])
#
#                 save_h5py(Name, features, class_sample, image_name, sample_path, question, masks, bbox)
#             bar()

# def process_text_embedding(text, device, policy_model):
#     # 这个函数需要根据您的具体模型来实现
#     # 这里只是一个示例
#     tokenized_text = policy_model.tokenizer(text, return_tensors="pt").to(device)
#     text_embeddings = policy_model.text_encoder(**tokenized_text).last_hidden_state.mean(dim=1)
#     return text_embeddings


def process_embedding(transform, image, device, policy_model):
    # image, _ = transform(image, image)
    image = image.to(device)
    embeddings = policy_model.preprocess_store(image.unsqueeze(0))  # len(input_list) x hidden_size
    return embeddings


def process_clip_embedding(preprocess, image, device, policy_model):
    image = preprocess(image).unsqueeze(0).to(device)
    another_embedding = policy_model.encode_image(image)
    embeddings = policy_model.encode_image(image).float()
    return embeddings

# def process_text_embedding(text, device, clip_model, preprocess):
#     text = clip.tokenize(text).to(device)
#     with torch.no_grad():
#         text_embeddings = clip_model.encode_text(text).float()
#     return text_embeddings
#
# def process_image_embedding(image, device, clip_model, preprocess):
#     processed_images = preprocess(image).to(device)
#     # 添加批次维度
#     processed_images = processed_images.unsqueeze(0)  # 新形状: (1, 3, 224, 224)
#     with torch.no_grad():
#         image_embeddings = clip_model.encode_image(processed_images).float()
#     return image_embeddings


def process_text_embedding(texts, device, clip_model, preprocess):
    # 确保texts是一个列表
    if not isinstance(texts, list):
        texts = [texts]

    # 批量处理文本
    tokenized_texts = clip.tokenize(texts).to(device)
    with torch.no_grad():
        text_embeddings = clip_model.encode_text(tokenized_texts).float()
    return text_embeddings


def process_image_embedding(images, device, clip_model, preprocess):
    # 确保images是一个列表
    if not isinstance(images, list):
        images = [images]

    # 批量处理图像
    processed_images = torch.stack([preprocess(image) for image in images]).to(device)
    with torch.no_grad():
        image_embeddings = clip_model.encode_image(processed_images).float()
    return image_embeddings


# def prepare_cluster_h5py(Name, all_classes, all_images, device, args):
#     with h5py.File(Name, 'a') as f:
#         with alive_bar(len(all_classes), title='CLUSTER···') as bar:
#             for voc_class in all_classes:
#                 all_boxes = []
#                 can_features = []
#                 can_images_name = []
#
#                 show_images_name = []  # cout element appear times
#                 if args.BENCHMARK != 'pascal_part':
#                     class_all_images = all_images[voc_class]
#                     voc_class = str(voc_class)
#                 else:
#                     class_all_images = all_images[voc_class]["images"]
#
#                 for image_path in class_all_images:
#                     image_name = os.path.basename(image_path)
#                     # image_name = os.path.basename(image_path["img_path"])
#
#                     if not f[voc_class][image_name].__contains__(str(0)):
#                         can_features.append(f[voc_class][image_name]['features'][:])
#                         can_images_name.append(image_name)
#                     else:
#                         now_chose_id = show_images_name.count(image_name)
#                         can_features.append(f[voc_class][image_name][str(now_chose_id)]['features'][:])
#                         can_images_name.append(image_name)
#
#                     show_images_name.append(image_name)
#
#
#                 can_features = np.array(can_features)
#
#                 if len(can_features.shape) > 2:
#                     can_features = can_features.squeeze(1)
#
#                 assert len(can_features.shape) == 2
#                 # can_features = np.array(can_features)
#                 cluster_id, cluster_centers = cluster_to_fix(can_features, device, args)
#
#                 # cluster_id = cluster_id.cpu().detach().numpy().astype(np.float32)
#                 cluster_centers = cluster_centers.cpu().detach().numpy().astype(np.float32)
#
#                 if f[voc_class].__contains__('cluster_result'):
#                     del f[voc_class]['cluster_result']
#
#                 subsub = f[voc_class]
#                 subsub_0 = subsub.create_group('cluster_result')
#                 subsub_0.create_dataset('all_image_name', data=can_images_name)
#                 subsub_0.create_dataset('cluster_id', data=cluster_id)
#                 subsub_0.create_dataset('cluster_centers', data=cluster_centers)
#
#                 bar()  # update cluster bar

def prepare_cluster_h5py(Name, all_classes, all_images, device, args):
    with h5py.File(Name, 'a') as f:
        with alive_bar(len(all_classes), title='CLUSTER···') as bar:
            for voc_class in all_classes:
                all_boxes = []
                can_features = []
                can_images_name = []
                can_questions = []  # 新增：存储问题

                show_images_name = []  # cout element appear times
                if args.BENCHMARK != 'pascal_part':
                    class_all_images = all_images[voc_class]
                    voc_class = str(voc_class)
                else:
                    class_all_images = all_images[voc_class]["images"]

                for image_item in class_all_images:
                    image_name = os.path.basename(image_item["img_path"])
                    question = image_item["question"]  # 获取问题

                    if not f[voc_class][image_name].__contains__(str(0)):
                        can_features.append(f[voc_class][image_name]['features'][:])
                        can_images_name.append(image_name)
                        can_questions.append(question)  # 添加问题
                    else:
                        now_chose_id = show_images_name.count(image_name)
                        can_features.append(f[voc_class][image_name][str(now_chose_id)]['features'][:])
                        can_images_name.append(image_name)
                        can_questions.append(question)  # 添加问题

                    show_images_name.append(image_name)

                can_features = np.array(can_features)

                if len(can_features.shape) > 2:
                    can_features = can_features.squeeze(1)

                assert len(can_features.shape) == 2
                cluster_id, cluster_centers = cluster_to_fix(can_features, device, args)

                cluster_centers = cluster_centers.cpu().detach().numpy().astype(np.float32)

                if f[voc_class].__contains__('cluster_result'):
                    del f[voc_class]['cluster_result']

                subsub = f[voc_class]
                subsub_0 = subsub.create_group('cluster_result')
                subsub_0.create_dataset('all_image_name', data=can_images_name)
                subsub_0.create_dataset('cluster_id', data=cluster_id)
                subsub_0.create_dataset('cluster_centers', data=cluster_centers)
                subsub_0.create_dataset('questions', data=can_questions)  # 新增：保存问题信息

                bar()  # update cluster bar


def prepare_cluster_no_category_h5py(Name, all_classes, all_images, device, args):
    with h5py.File(Name, 'a') as f:
        all_boxes = []
        can_features = []
        can_images_name = []
        can_images_class = []
        for voc_class in all_classes:
            show_images_name = []  # cout element appear times
            class_all_images = all_images[voc_class]
            voc_class = str(voc_class)
            for image_path in class_all_images:
                image_name = os.path.basename(image_path)
                if not f[voc_class][image_name].__contains__(str(0)):
                    can_features.append(f[voc_class][image_name]['features'][:])
                    can_images_name.append(image_name)
                else:
                    now_chose_id = show_images_name.count(image_name)
                    can_features.append(f[voc_class][image_name][str(now_chose_id)]['features'][:])
                    can_images_name.append(image_name)

                show_images_name.append(image_name)
                can_images_class.append(voc_class)

        can_features = np.array(can_features).squeeze(1)

        with alive_bar(1, title='CLUSTER···') as bar:
            cluster_id, cluster_centers = cluster_to_fix(can_features, device, args)
            cluster_centers = cluster_centers.cpu().detach().numpy().astype(np.float32)
            if f.__contains__('cluster_result'):
                del f['cluster_result']
            subsub_0 = f.create_group('cluster_result')
            subsub_0.create_dataset('all_image_name', data=can_images_name)
            subsub_0.create_dataset('cluster_id', data=cluster_id)
            subsub_0.create_dataset('cluster_centers', data=cluster_centers)
            subsub_0.create_dataset('all_image_class', data=can_images_class)

            bar()  # update cluster bar


def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch


def to_cpu(tensor):
    return tensor.detach().clone().cpu()


def is_integer(number):
    number = re.sub(r"(?<=\d)[\,](?=\d)", "", number)  # "12,456" -> "12456"
    if re.findall(r"^-?[\d]+$", number):
        return True
    else:
        return False


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def convert_table_text_to_pandas(table_text):
    _data = {}

    table_text = re.sub(r" ?\| ?", " | ", table_text)
    cells = [row.split(" | ") for row in table_text.split("\n")]

    row_num = len(cells)
    column_num = len(cells[0])

    # for table without a header
    first_row = cells[0]
    matches = re.findall(r"[\d]+", " ".join(first_row))
    if len(matches) > 0:
        header = [f"Column {i+1}" for i in range(column_num)]
        cells.insert(0, header)

    # build DataFrame for the table
    for i in range(column_num):
        _data[cells[0][i]] = [row[i] for row in cells[1:]]

    table_pd = pd.DataFrame.from_dict(_data)

    return table_pd


class Logger(object):

    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(str(msg) + '\n')
        self.log_file.flush()
        print(msg)


def prepare_cand_h5py(Name, all_classes, all_images, args):
    with h5py.File(Name, 'a') as f:
        with alive_bar(len(all_classes), title='CLUSTER···') as bar:
            for voc_class in all_classes:
                can_features = []
                can_images_name = []

                show_images_name = []  # cout element appear times
                class_all_images = all_images[voc_class]
                voc_class = str(voc_class)

                for image_path in class_all_images:
                    image_name = os.path.basename(image_path["img_path"])


                    if not f[voc_class][image_name].__contains__(str(0)):
                        can_features.append(f[voc_class][image_name]['features'][:])
                        can_images_name.append(image_name)
                    else:
                        now_chose_id = show_images_name.count(image_name)
                        can_features.append(f[voc_class][image_name][str(now_chose_id)]['features'][:])
                        can_images_name.append(image_name)
                    show_images_name.append(image_name)

                can_features = np.array(can_features)

                if len(can_features.shape) > 2:
                    can_features = can_features.squeeze(1)

                assert len(can_features.shape) == 2
                # can_features = np.array(can_features)
                device = torch.device('cpu')
                selected_labels, selected_centers = cluster_to_fix(can_features, device, args)

                selected_inds = []
                for now_label in range(len(selected_centers)):
                    sub_cluster_ids = np.where(selected_labels == now_label)[0]
                    sub_cluster_features = [can_features[now_id] for now_id in sub_cluster_ids]
                    # selected_inds += cluster_nearest_farthest(selected_centers[now_label], sub_cluster_features)
                    selected_inds += cluster_nearest_farthest2(selected_centers[now_label], sub_cluster_features)

                selected_centers = selected_centers.cpu().detach().numpy().astype(np.float32)
                selected_features = [can_features[selected_id] for selected_id in selected_inds]
                selected_image_name = [can_images_name[selected_id] for selected_id in selected_inds]

                if f[voc_class].__contains__('cluster_result'):
                    del f[voc_class]['cluster_result']

                if f[voc_class].__contains__('selected_result'):
                    del f[voc_class]['selected_result']

                subsub = f[voc_class]
                subsub_0 = subsub.create_group('selected_result')
                subsub_0.create_dataset('selected_image_name', data=selected_image_name)
                subsub_0.create_dataset('selected_centers', data=selected_centers)
                subsub_0.create_dataset('selected_features', data=selected_features)



                bar()  # update cluster bar

