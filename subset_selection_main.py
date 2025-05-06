import os
import pickle
import numpy as np
from feature_extraction_utils import (
    get_video_files, get_image_files, extract_clip_features_three_frames,
    extract_actionclip_features, extract_clip_image_feature, extract_clip_text_feature,
    ActionCLIPFeatureExtractor
)
import clip
import torch
from submodlib.functions import GraphCutMutualInformationFunction

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

def extract_and_save_features(folder_path, method, output_pkl):
    video_files = get_video_files(folder_path)
    video_names = []
    feature_list = []
    if method in ["clip_mean", "clip_concat"]:
        clip_model, preprocess = clip.load("ViT-B/16", device=device)
        for video_path in video_files:
            print(f"Extracting features from {video_path}")
            feat = extract_clip_features_three_frames(
                video_path, clip_model, preprocess,
                mode="mean" if method=="clip_mean" else "concat"
            )
            if feat is not None:
                feature_list.append(feat)
                video_names.append(video_path)
            else:
                print(f"Warning: Could not extract features from {video_path}")
    elif method == "actionclip":
        actionclip_extractor = ActionCLIPFeatureExtractor(device=device)
        for video_path in video_files:
            print(f"Extracting features from {video_path}")
            feat = extract_actionclip_features(video_path, actionclip_extractor)
            if feat is not None:
                feature_list.append(feat)
                video_names.append(video_path)
            else:
                print(f"Warning: Could not extract features from {video_path}")
    else:
        raise ValueError("method must be 'clip_mean', 'clip_concat', or 'actionclip'")
    if feature_list:
        feature_matrix = np.stack(feature_list, axis=0)
        with open(output_pkl, "wb") as f:
            pickle.dump({"features": feature_matrix, "video_paths": video_names}, f)
        print(f"Saved feature matrix and video paths to {output_pkl}")
    else:
        print("No features extracted from any video.")

def extract_query_feature(query_type, query_value, method):
    if method in ["clip_mean", "clip_concat"]:
        clip_model, preprocess = clip.load("ViT-B/16", device=device)
        if query_type == "text":
            return extract_clip_text_feature(query_value, clip_model)
        elif query_type == "image":
            return extract_clip_image_feature(query_value, clip_model, preprocess)
        elif query_type == "video":
            return extract_clip_features_three_frames(query_value, clip_model, preprocess, mode="mean")
        else:
            raise ValueError("query_type must be 'text', 'image', or 'video'")
    elif method == "actionclip":
        if query_type == "video":
            actionclip_extractor = ActionCLIPFeatureExtractor(device=device)
            return extract_actionclip_features(query_value, actionclip_extractor)
        else:
            raise ValueError("ActionCLIP only supports video queries in this setup.")
    else:
        raise ValueError("method must be 'clip_mean', 'clip_concat', or 'actionclip'")

def subset_selection(feature_matrix, query_feature, num_select=10):
    n, d = feature_matrix.shape
    query_feature = query_feature.reshape(1, -1)
    obj = GraphCutMutualInformationFunction(
        n=n,
        num_queries=1,
        data=feature_matrix.astype(np.float32),
        queryData=query_feature.astype(np.float32),
        metric="cosine"
    )
    greedyList = obj.maximize(budget=num_select, optimizer="NaiveGreedy", stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    selected_indices = sorted([x[0] for x in greedyList])
    return selected_indices

def main():
    # --- USER CONFIGURABLE ---
    folder_path = "/home/ramana/Video semantic output"  # Path to videos
    method = "clip_mean"  # 'clip_mean', 'clip_concat', or 'actionclip'
    feature_pkl = "video_features.pkl"  # Where to store features
    # Query can be text, image, or video
    query_type = "text"  # 'text', 'image', or 'video'
    query_value = "Weapons"  # e.g., "Weapons" or "/path/to/query.jpg" or "/path/to/query.mp4"
    num_select = 10  # Number of videos to select
    subset_pkl = "selected_subset.pkl"  # Where to store selected subset
    # --- END USER CONFIGURABLE ---

    # 1. Extract and save features
    extract_and_save_features(folder_path, method, feature_pkl)

    # 2. Load features
    with open(feature_pkl, "rb") as f:
        data = pickle.load(f)
    feature_matrix = data["features"]
    video_paths = data["video_paths"]

    # 3. Extract query feature
    query_feature = extract_query_feature(query_type, query_value, method)

    # 4. Subset selection
    selected_indices = subset_selection(feature_matrix, query_feature, num_select=num_select)
    selected_paths = [video_paths[i] for i in selected_indices]
    print("Selected subset:")
    for p in selected_paths:
        print(p)
    # 5. Save selected subset as pickle
    with open(subset_pkl, "wb") as f:
        pickle.dump(selected_paths, f)
    print(f"Saved selected subset paths to {subset_pkl}")

if __name__ == "__main__":
    main()
