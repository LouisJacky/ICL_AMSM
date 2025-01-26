import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

class MultiModalMatchingModel(nn.Module):
    def __init__(self, clip_model, hidden_size=512):
        super(MultiModalMatchingModel, self).__init__()
        self.clip_model = clip_model
        self.hidden_size = hidden_size

        # 图像和文本特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(clip_model.visual.output_dim + clip_model.text_projection.shape[1], hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )

        # 查询和示例特征比较层
        self.comparison_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),  # 保留这一层
            nn.LayerNorm(hidden_size // 4),  # 使用LayerNorm替代BatchNorm1d
            nn.GELU(),  # 使用GELU替代LeakyReLU
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, query_images, query_questions, example_images, example_questions):
        query_fused = self.encode_image_text(query_images, query_questions)
        example_fused = self.encode_image_text(example_images, example_questions)
        return self.compute_similarity(query_fused, example_fused)

    def encode_image_text(self, images, questions):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images).float().detach()
            text_features = self.clip_model.encode_text(clip.tokenize(questions).to(images.device)).float().detach()
        combined_features = torch.cat([image_features, text_features], dim=1)
        return self.fusion_layer(combined_features)

    def compute_similarity(self, query_features, example_features):
        combined_features = torch.cat([query_features, example_features], dim=1)
        return self.comparison_layer(combined_features).squeeze()

    def batch_compute_similarity(self, query_features, example_features):
        query_features_expanded = query_features.unsqueeze(1)
        example_features_expanded = example_features.unsqueeze(0)
        combined_features = torch.cat([
            query_features_expanded.expand(-1, example_features.size(0), -1),
            example_features_expanded.expand(query_features.size(0), -1, -1)
        ], dim=2)
        return self.comparison_layer(combined_features.view(-1, self.hidden_size * 2)).view(query_features.size(0), -1)

class AdaptiveMultiModalMatchingModel(nn.Module):
    def __init__(self, clip_model, hidden_size=512, Modal='both'):
        super(AdaptiveMultiModalMatchingModel, self).__init__()
        self.clip_model = clip_model
        self.hidden_size = hidden_size
        self.Modal=Modal

        # 图像和文本特征融合层
        if self.Modal=='both':
            self.fusion_layer = nn.Sequential(
                nn.Linear(clip_model.visual.output_dim + clip_model.text_projection.shape[1], hidden_size * 2),
                nn.LayerNorm(hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU()
            )
            # self.projector = nn.Sequential(
            #     nn.Linear(clip_model.text_projection.shape[1], clip_model.text_projection.shape[1]),
            #     nn.LayerNorm(clip_model.text_projection.shape[1]),
            #     nn.GELU(),
            # )

        elif self.Modal=='text':
            self.fusion_layer = nn.Sequential(
                nn.Linear(clip_model.text_projection.shape[1], hidden_size * 2),
                nn.LayerNorm(hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU()
            )
        elif self.Modal=='image':
            self.fusion_layer = nn.Sequential(
                nn.Linear(clip_model.visual.output_dim, hidden_size * 2),
                nn.LayerNorm(hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU()
            )

        # 查询和示例特征比较层
        self.comparison_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),  # 保留这一层
            nn.LayerNorm(hidden_size // 4),  # 使用LayerNorm替代BatchNorm1d
            nn.GELU(),  # 使用GELU替代LeakyReLU
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, query_images=None, query_questions=None, example_images=None, example_questions=None):
        query_fused = self.encode_image_text(query_images, query_questions)
        example_fused = self.encode_image_text(example_images, example_questions)
        return self.compute_similarity(query_fused, example_fused)

    def encode_image_text(self, images=None, questions=None):
        if self.Modal == 'both':
            with torch.no_grad():
                image_features = self.clip_model.encode_image(images).float().detach()
                text_features = self.clip_model.encode_text(clip.tokenize(questions).to(images.device)).float().detach()
            # text_features_projected = self.projector(text_features)
            # combined_features = torch.cat([image_features, text_features_projected], dim=1)
            combined_features = torch.cat([image_features, text_features], dim=1)
        elif self.Modal == 'text':
            with torch.no_grad():
                text_features = self.clip_model.encode_text(clip.tokenize(questions).to(images.device)).float().detach()
            combined_features = text_features
        elif self.Modal == 'image':
            with torch.no_grad():
                image_features = self.clip_model.encode_image(images).float().detach()
            combined_features = image_features
        return self.fusion_layer(combined_features)

    def compute_similarity(self, query_features, example_features):
        combined_features = torch.cat([query_features, example_features], dim=1)
        return self.comparison_layer(combined_features).squeeze()

    def batch_compute_similarity(self, query_features, example_features):
        query_features_expanded = query_features.unsqueeze(1)
        example_features_expanded = example_features.unsqueeze(0)
        combined_features = torch.cat([
            query_features_expanded.expand(-1, example_features.size(0), -1),
            example_features_expanded.expand(query_features.size(0), -1, -1)
        ], dim=2)
        return self.comparison_layer(combined_features.view(-1, self.hidden_size * 2)).view(query_features.size(0), -1)


class ResCaptionMatchingModel(nn.Module):
    def __init__(self, clip_model, hidden_size=512):
        super(ResCaptionMatchingModel, self).__init__()
        self.clip_model = clip_model
        self.hidden_size = hidden_size

        # 图像和文本特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(clip_model.visual.output_dim, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )

        # 查询和示例特征比较层
        self.comparison_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),  # 保留这一层
            nn.LayerNorm(hidden_size // 4),  # 使用LayerNorm替代BatchNorm1d
            nn.GELU(),  # 使用GELU替代LeakyReLU
            nn.Linear(hidden_size // 4, 1),
        )

    def forward(self, query_images, example_images):
        query_fused = self.encode_image(query_images)
        example_fused = self.encode_image(example_images)
        return self.compute_similarity(query_fused, example_fused)

    def encode_image(self, images):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images).float().detach()
        return self.fusion_layer(image_features)

    def compute_similarity(self, query_features, example_features):
        combined_features = torch.cat([query_features, example_features], dim=1)
        return self.comparison_layer(combined_features).squeeze()

    def batch_compute_similarity(self, query_features, example_features):
        query_features_expanded = query_features.unsqueeze(1)
        example_features_expanded = example_features.unsqueeze(0)
        combined_features = torch.cat([
            query_features_expanded.expand(-1, example_features.size(0), -1),
            example_features_expanded.expand(query_features.size(0), -1, -1)
        ], dim=2)
        return self.comparison_layer(combined_features.view(-1, self.hidden_size * 2)).view(query_features.size(0), -1)


class ResMultiModalMatchingPolicyModel(nn.Module):
    def __init__(self, clip_model, hidden_size=512):
        super(ResMultiModalMatchingPolicyModel, self).__init__()
        self.clip_model = clip_model
        self.hidden_size = hidden_size

        # 图像特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(clip_model.visual.output_dim + clip_model.text_projection.shape[1], hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )

        # 使用与policy_network相同的projector和predictor结构
        self.projector = self._build_mlp(3, hidden_size, hidden_size, hidden_size)
        self.predictor = self._build_mlp(2, hidden_size, hidden_size, hidden_size)

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def forward(self, query_images, query_questions, example_images, example_questions):
        query_fused = self.encode_image_text(query_images, query_questions)
        example_fused = self.encode_image_text(example_images, example_questions)
        return self.compute_similarity(query_fused, example_fused)

    def encode_image_text(self, images, questions):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(images).float().detach()
            text_features = self.clip_model.encode_text(clip.tokenize(questions).to(images.device)).float().detach()
        combined_features = torch.cat([image_features, text_features], dim=1)
        return self.fusion_layer(combined_features)

    def compute_similarity(self, query_features, example_features):
        # 使用projector和predictor计算相似度
        query_proj = query_features + self.projector(query_features)
        example_proj = example_features + self.projector(example_features)

        example_pred = self.predictor(example_proj)
        similarity = F.cosine_similarity(query_proj, example_pred, dim=1)
        return similarity

    def batch_compute_similarity(self, query_features, example_features):
        # 批量计算相似度
        query_proj = query_features + self.projector(query_features)
        example_proj = example_features + self.projector(example_features)

        example_pred = self.predictor(example_proj)

        # 计算批量余弦相似度
        query_norm = F.normalize(query_proj, p=2, dim=1)
        example_norm = F.normalize(example_pred, p=2, dim=1)
        similarity = torch.mm(query_norm, example_norm.t())

        return similarity

