import os
import sys
import pickle
import pandas as pd

from .model_apis import *

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
print("DEVICE:", DEVICE)

# ============= helper functions ================================================
def load_model(model_name='stats', args=None):
    # all models should follows the function structure of AST_API
    if model_name == 'clap':
        model = CLAP_API()
    elif model_name == 'normwear':
        model = NormWear_API(weight_path=args.model_weight_dir)
    elif model_name == 'chronos':
        model = Chronos_API()
    # added baselines
    elif model_name == 'stats':
        model = STAT_API()
    elif model_name == 'tfc':
        model = TFC_API()
    elif model_name == 'demo':
        model = Demogr_API()
    elif model_name == 'crossvit':
        model = CrossVitAPI()
    else:
        print("Model not supported. ")
        exit()
    
    # return
    model = model.to(DEVICE)
    model.eval()

    # # check number of parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"{model_name} Number of parameters: {total_params}")
    # exit()

    return model

def audio_embedding_prepare(data_rootpath="audio_downstream/Coswara", model_name='ast', root_prefix="../", remark="", args=None):
    # construct model
    model = load_model(model_name=model_name, args=args)

    save_remark = remark if len(remark) > 0 else model_name

   

    # resolve paths from this file location (instead of root_prefix)
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)  # .../NormWear
    base_data_dir = os.path.normpath(os.path.join(project_root, "data", data_rootpath))
    sample_dir = os.path.join(base_data_dir, "sample_for_downstream")
    embed_dir = os.path.join(base_data_dir, f"{save_remark}_wav_embed")

    # initialize folder for save all the embedding
    os.makedirs(embed_dir, exist_ok=True)

    # get embedding for each sample
    for fn in tqdm(sorted(os.listdir(sample_dir))):
        # edge case
        if fn[0] == '.':
            continue
        
        # load sample
        # read data
        with open(os.path.join(sample_dir, fn), 'rb') as f:
            sample = pickle.load(f) # ['uid', 'data', 'label', 'sampling_rate']
        
        # expand 1 dimension if only single dimension
        if len(sample['data'].shape) == 1:
            sample['data'] = np.expand_dims(sample['data'], axis=0)

        # test clap pipeline
        with torch.no_grad():
            # if model_name in["demo", "normwear"]:
            if model_name in["demo"]: # normal demographic test
                embed = model.get_embedding(
                    sample['data'], 
                    sampling_rate=sample['sampling_rate'],
                    device=DEVICE,
                    sub_info=(data_rootpath, fn, root_prefix) # comment out if not demographic
                ) # E
            else:
                embed = model.get_embedding(
                    sample['data'], 
                    sampling_rate=sample['sampling_rate'],
                    device=DEVICE,
                ) # E

        # # check
        # print(embed.shape)
        # print(torch.mean(embed))
        # # print(sample['label'])
        # exit()
        
        # save the embedding
        with open(os.path.join(embed_dir, fn), 'wb') as f:
            pickle.dump({
                "uid": sample["uid"], 
                "sampling_rate": sample['sampling_rate'], 
                "embed": embed.cpu().numpy().astype(np.float16), # E
                "label": sample['label']
            }, f)

def combine_normwear_ast(data_rootpath="audio_downstream/Coswara", root_prefix="../"):
    base_data_dir = os.path.normpath(os.path.join(root_prefix, "data", data_rootpath))
    sample_dir = os.path.join(base_data_dir, "sample_for_downstream")
    normwear_dir = os.path.join(base_data_dir, "normwear_wav_embed")
    ast_dir = os.path.join(base_data_dir, "ast_wav_embed")
    combine_dir = os.path.join(base_data_dir, "nacombine_wav_embed")

    # initialize folder for save all the embedding
    os.makedirs(combine_dir, exist_ok=True)

    # get embedding for each sample
    for fn in tqdm(sorted(os.listdir(sample_dir))):
        # edge case
        if fn[0] == '.':
            continue
        
        # load sample
        # read data
        with open(os.path.join(sample_dir, fn), 'rb') as f:
            sample = pickle.load(f) # ['uid', 'data', 'label', 'sampling_rate']
        
        # TODO combine embeds
        with open(os.path.join(normwear_dir, fn), 'rb') as f:
            normwear_embed = pickle.load(f)["embed"]
        with open(os.path.join(ast_dir, fn), 'rb') as f:
            ast_embed = pickle.load(f)["embed"]
        embed = np.concatenate((normwear_embed, ast_embed), axis=0)

        # # check
        # print(embed.shape)
        # print(np.mean(embed), np.std(embed), np.min(embed), np.max(embed))
        # exit()
        
        # save the embedding
        with open(os.path.join(combine_dir, fn), 'wb') as f:
            pickle.dump({
                "uid": sample["uid"], 
                "sampling_rate": sample['sampling_rate'], 
                "embed": embed, # E*2
                "label": sample['label']
            }, f)

if __name__ == '__main__':
    # python3 -m src.downstream.prepare_embeds chronos audio_downstream/KAUH
    # python3 -m src.downstream.prepare_embeds ast downstream/PPG_DM

    # input model name
    model_name = sys.argv[1] # ast, clap, opera, normwear
    data_rootpath = sys.argv[2] # e.g. audio_downstream/KAUH

    # process to get all embeds
    audio_embedding_prepare(model_name=model_name, data_rootpath=data_rootpath)

    # # combine embeds
    # # python3 -m src.downstream.prepare_embeds audio_downstream/KAUH
    # data_rootpath = sys.argv[1] # e.g. audio_downstream/KAUH
    # combine_normwear_ast(data_rootpath=data_rootpath)