import json
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import numpy as np

class VQADataset(Dataset):
    def __init__(
        self, image_dir_path, question_path, annotations_path, is_train, dataset_name, transform
    ):
        self.transform = transform

        self.questions = json.load(open(question_path, "r"))["questions"]
        if annotations_path is not None:
            self.answers = json.load(open(annotations_path, "r"))["annotations"]
        else:
            self.answers = None
        self.image_dir_path = image_dir_path
        self.is_train = is_train
        self.dataset_name = dataset_name
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            self.img_coco_split = self.image_dir_path.strip("/").split("/")[-1]
            assert self.img_coco_split in {"train2014", "val2014", "test2015"}
        if self.dataset_name in {"vqav2"} and self.is_train == False:
            retrieval_path = "retrieval_results/vqav2/validation_sim_32.npy"
            self.retrieval_set = np.load(retrieval_path, allow_pickle=True).item()
        if self.dataset_name in {"ok_vqa"} and self.is_train == False:
            retrieval_path = "retrieval_results/okvqa_validation_SQQR.npy"
            self.retrieval_set = np.load(retrieval_path, allow_pickle=True).item()
        # if self.dataset_name in {"vizwiz"} and self.is_train == False:
        #     retrieval_path = "retrieval_results/vizwiz_validation_SQQR.npy"
        #     self.retrieval_set = np.load(retrieval_path, allow_pickle=True).item()
        if self.dataset_name in {"vizwiz"} and self.is_train == False:
            retrieval_path = "/path/to/Code/ICL_diversity_ofv2/retrieval_results/vizwiz_validation_SQ.npy"
            self.retrieval_set = np.load(retrieval_path, allow_pickle=True).item()

        # 添加 img_metadata 和 img_metadata_classwise
        self.img_metadata = []
        self.img_metadata_classwise = {}

        for question in self.questions:
            img_path = self.get_img_path(question)
            self.img_metadata.append(img_path)

            # # 使用 cluster 类别 (这只是一个示例,你可能需要根据实际情况调整)
            # class_sample = question["cluster"]
            #
            # if class_sample not in self.img_metadata_classwise:
            #     self.img_metadata_classwise[class_sample] = [img_path]
            # else:
            #     self.img_metadata_classwise[class_sample].append(img_path)

            # 使用 cluster 类别
            class_sample = question["cluster"]

            # 创建包含 img_path 和问题的字典
            item = {
                "img_path": img_path,
                "question": question["question"]
            }

            if class_sample not in self.img_metadata_classwise:
                self.img_metadata_classwise[class_sample] = [item]
            else:
                self.img_metadata_classwise[class_sample].append(item)

        # 添加一个字典来存储图像路径到样本的映射
        self.samples_by_path = {}
        for idx, question in enumerate(self.questions):
            img_path = self.get_img_path(question)
            self.samples_by_path[img_path] = idx

    def __len__(self):
        return len(self.questions)

    def id2item(self, idx):
        question = self.questions[idx]
        answers = self.answers[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)
        return {
            "image": image,
            "image_id": question['image_id'],
            "question": question["question"],
            "answers": [a["answer"] for a in answers["answers"]],
            "question_id": question["question_id"],
        }

    def get_img_path(self, question):
        if self.dataset_name in {"vqav2", "ok_vqa"}:
            return os.path.join(
                self.image_dir_path,
                f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg"
                if self.is_train
                else f"COCO_{self.img_coco_split}_{question['image_id']:012d}.jpg",
            )
        elif self.dataset_name == "vizwiz":
            return os.path.join(self.image_dir_path, question["image_id"])
        elif self.dataset_name == "textvqa":
            return os.path.join(self.image_dir_path, f"{question['image_id']}.jpg")
        else:
            raise Exception(f"Unknown VQA dataset {self.dataset_name}")

    def __getitem__(self, idx):
        question = self.questions[idx]
        img_path = self.get_img_path(question)
        image = Image.open(img_path)

        query_img = self.transform(image)

        image.load()
        results = {
            "image": image,
            "query_img": query_img,
            "image_id": question['image_id'],
            "question": question["question"],
            "question_id": question["question_id"],
            'class_id': question["cluster"],
        }
        if self.answers is not None:
            answers = self.answers[idx]
            results["answers"] = [a["answer"] for a in answers["answers"]]
        # if you need other retrieval method, add it into the results.
        if self.dataset_name in {"vqav2"} and self.is_train == False:
            question_id = question['question_id']
            results["SI"] = [i[2] for i in self.retrieval_set[question_id]["SI"]]
            results["SI_Q"] = [i[2] for i in self.retrieval_set[question_id]["SI_Q"]]
            results["SQ"] = [i[2] for i in self.retrieval_set[question_id]["SQ"]]
            results["SQ_I"] = [i[2] for i in self.retrieval_set[question_id]["SQ_I"]]
            results["SI_1"] = [i[2] for i in self.retrieval_set[question_id]["SI_1"]]
            results["SI_2"] = [i[2] for i in self.retrieval_set[question_id]["SI_2"]]
        if self.dataset_name in {"ok_vqa"} and self.is_train == False:
            question_id = question['question_id']
            results["SQ"] = [i[2] for i in self.retrieval_set[question_id]["SQ"]]
            results["SQ_I"] = [i[2] for i in self.retrieval_set[question_id]["SQ_I"]]
        if self.dataset_name in {"vizwiz"} and self.is_train == False:
            question_id = question['question_id']
            results["SQ"] = [i[2] for i in self.retrieval_set[question_id]["SQ"]]
            # results["SQ_I"] = [i[2] for i in self.retrieval_set[question_id]["caption_image"]]
            results["SQ_I"] = [i[2] for i in self.retrieval_set[question_id]["SQ_I"]]
        return results

    def get_sample_by_image_path(self, image_path):
        if image_path in self.samples_by_path:
            idx = self.samples_by_path[image_path]
            question = self.questions[idx]
            result = {
                'image_path': image_path,
                'question': question['question'],
                'question_id': question['question_id'],
                'image_id': question['image_id'],
            }
            if self.answers is not None:
                answers = self.answers[idx]
                result['answers'] = [a['answer'] for a in answers['answers']]
            return result
        return None

