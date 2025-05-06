import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import clip

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

VIDEO_EXTS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp']

# --- ActionCLIP Transformer Classes ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.projection = nn.Linear(embed_dim, embed_dim)
    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x_mean = torch.mean(x, dim=1)
        output = self.projection(x_mean)
        return output, x

class ActionCLIPFeatureExtractor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, self.preprocess = clip.load("ViT-B/16", device=self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        embed_dim = self.clip_model.visual.output_dim
        self.transformer = SpatioTemporalTransformer(
            embed_dim=embed_dim,
            num_heads=8,
            num_layers=4,
            dropout=0.1
        ).to(self.device)
        self.clip_model.eval()
        self.transformer.eval()
    def extract_frames(self, video_path, max_frames=32):
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if frame_count <= 0 or np.isnan(fps) or fps <= 0:
            frame_interval = 10
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % frame_interval == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame)
                    preprocessed_img = self.preprocess(pil_img)
                    frames.append(preprocessed_img)
                idx += 1
            cap.release()
            if frames:
                sample_indices = np.linspace(0, len(frames)-1, min(max_frames, len(frames)), dtype=int)
                frames = [frames[i] for i in sample_indices]
            else:
                raise ValueError("No frames could be extracted from the video")
        else:
            frame_indices = np.linspace(0, frame_count-1, min(max_frames, frame_count), dtype=int)
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame)
                    preprocessed_img = self.preprocess(pil_img)
                    frames.append(preprocessed_img)
            cap.release()
            if not frames:
                raise ValueError("No frames could be extracted from the video")
        frames_tensor = torch.stack(frames)
        return frames_tensor
    def extract_clip_features(self, frames_tensor, batch_size=8):
        num_frames = frames_tensor.shape[0]
        frame_features = []
        with torch.no_grad():
            for i in range(0, num_frames, batch_size):
                batch = frames_tensor[i:i+batch_size].to(self.device)
                features = self.clip_model.encode_image(batch)
                features = F.normalize(features, dim=-1)
                features = features.cpu()
                frame_features.append(features)
        all_features = torch.cat(frame_features, dim=0)
        return all_features
    def process_video(self, video_path):
        try:
            frames = self.extract_frames(video_path)
            clip_features = self.extract_clip_features(frames)
            with torch.no_grad():
                clip_features = clip_features.to(self.device).unsqueeze(0)
                global_features, all_features = self.transformer(clip_features)
                global_features = global_features.cpu().numpy()
                all_features = all_features.cpu().numpy()
            return {
                "global_features": global_features,  # [1, 512]
                "spatio_temporal_features": all_features,  # [1, num_frames, 512]
                "num_frames": frames.shape[0]
            }
        except Exception as e:
            print(f"Error processing video: {e}")
            return None

def get_video_files(folder_path):
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTS
    ]

def get_image_files(folder_path):
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]

def extract_clip_features_three_frames(video_path, clip_model, preprocess, mode="mean"):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return None
    indices = [0, frame_count // 2, frame_count - 1]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
        frames.append(preprocess(pil_img))
    cap.release()
    if len(frames) < 1:
        return None
    frames_tensor = torch.stack(frames).to(device)
    with torch.no_grad():
        features = clip_model.encode_image(frames_tensor)
        features = torch.nn.functional.normalize(features, dim=-1)
    if mode == "mean":
        return features.mean(dim=0).cpu().numpy()  # [512]
    elif mode == "concat":
        return features.flatten().cpu().numpy()    # [1536]
    else:
        raise ValueError("mode must be 'mean' or 'concat'")

def extract_clip_image_feature(image_path, clip_model, preprocess):
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = clip_model.encode_image(img_tensor)
        feat = torch.nn.functional.normalize(feat, dim=-1)
    return feat.squeeze().cpu().numpy()  # [512]

def extract_clip_text_feature(text, clip_model):
    with torch.no_grad():
        tokens = clip.tokenize([text]).to(device)
        feat = clip_model.encode_text(tokens)
        feat = torch.nn.functional.normalize(feat, dim=-1)
    return feat.squeeze().cpu().numpy()  # [512]

def extract_actionclip_features(video_path, actionclip_extractor):
    features = actionclip_extractor.process_video(video_path)
    if features and "global_features" in features:
        return features["global_features"].squeeze()  # [512]
    return None
