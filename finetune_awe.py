import torch
from transformers import AutoFeatureExtractor, Wav2Vec2Model,AutoTokenizer
from os import rename
from datasets import load_dataset, load_metric,concatenate_datasets,Dataset
import pandas as pd
from torch.utils.data import DataLoader
import os
import librosa
from scipy import spatial
from scipy.io.wavfile import write
import numpy as np
from sklearn.metrics import accuracy_score
import contrastiveLossELI5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
# -*- coding: utf-8 -*-
def my_collate(batch):
    data = [item["audio_1"] for item in batch]
    target = [item["audio_2"] for item in batch]
    # target = torch.LongTensor(target)
    return [data, target]
model_checkpoint = "facebook/wav2vec2-base"
batch_size = 1
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_checkpoint
)
model = Wav2Vec2Model.from_pretrained(model_checkpoint).to(device)
dataset_name = "buckeye"
#if buckeye.csv exists, load it as pandas dataframe
if os.path.exists("buckeye.csv"):
    df = pd.read_csv("buckeye.csv")
    print("loaded buckeye.csv")
else:
    #traverse the folder and get all the wav files stored in a pandas dataframe
    df = pd.DataFrame(columns=['audio_path','speaker_id','audio_id','audio'])
    for root, dirs, files in os.walk("/nethome/mmshaik/thesis/finetune_awe/data/"):
        for file in files:
            if file.endswith(".wav"):
                df = df.append({'audio_path': os.path.join("/".join(root.split('/')[-2:]), file),'speaker_id':file.split('.')[0][0:3],'audio_id': file.split('.')[0][3:],'audio':librosa.load(os.path.join(root, file),sr=16000)[0]}, ignore_index=True)
                
    df.to_csv('buckeye.csv',index=False)
#read the list file buckeye.utd_pairs.list
if os.path.exists("buckeye_pairs.pkl"):
    #read pickle file
    df_pairs = pd.read_pickle("buckeye_pairs.pkl")
    print("loaded buckeye_pairs.pkl")
else:
    df_pairs = pd.DataFrame(columns=['audio_1','audio_2','audio_1_length','audio_2_length'])
    with open('buckeye.utd_pairs.list') as f:
        utd_pairs = f.readlines()
        for i,utd_pair in enumerate(utd_pairs):
            if i > 14000:
                break
            if "s04_03b" in utd_pair or "s04_04a" in utd_pair:
                continue
            audio_path_pair_1 = ''.join(['data/',utd_pair.strip().split(' ')[1].split('_')[0],'/',''.join(utd_pair.strip().split(' ')[1].split('_')),'.wav'])
            start_audio_1 = utd_pair.strip().split(' ')[2]
            stop_audio_1 = utd_pair.strip().split(' ')[3]
            audio_path_pair_2 = ''.join(['data/',utd_pair.strip().split(' ')[4].split('_')[0],'/',''.join(utd_pair.strip().split(' ')[4].split('_')),'.wav'])
            start_audio_2 = utd_pair.strip().split(' ')[-2]
            stop_audio_2 = utd_pair.strip().split(' ')[-1]
            start_audio_1 = int(round((int(start_audio_1)*16000)/100))
            stop_audio_1 = int(round((int(stop_audio_1)*16000)/100))
            start_audio_2 = int(round((int(start_audio_2)*16000)/100))
            stop_audio_2 = int(round((int(stop_audio_2)*16000)/100))
            #read only particualr time of audio using librosa
            print(f"1:{df.loc[df['audio_path'] == audio_path_pair_1, 'audio_path'].to_string(index=False)}")
            print(f"2:{df.loc[df['audio_path'] == audio_path_pair_2, 'audio_path'].to_string(index=False)}")
            print(audio_path_pair_2)
            print(utd_pair)
            audio_1 = librosa.load(df.loc[df['audio_path'] == audio_path_pair_1, 'audio_path'].to_string(index=False),sr=16000)[0][int(start_audio_1):int(stop_audio_1)]
            #save the audio in a folder
            # write('1.wav', 16000,audio_1)
            audio_2 = librosa.load(df.loc[df['audio_path'] == audio_path_pair_2, 'audio_path'].to_string(index=False),sr=16000)[0][int(start_audio_2):int(stop_audio_2)]
            # write('2.wav', 16000,audio_2)
            audio_1_length = model._get_feat_extract_output_lengths(len(audio_1)).numpy()
            audio_2_length = model._get_feat_extract_output_lengths(len(audio_2)).numpy()
            df_pairs = df_pairs.append({'audio_1':audio_1,'audio_2':audio_2,'audio_1_length':audio_1_length.reshape(1),'audio_2_length':audio_2_length.reshape(1)},ignore_index=True)
    df_pairs.to_pickle('buckeye_pairs.pkl')        
    # df_pairs.to_csv('buckeye_pairs.csv',index=False)


dset = Dataset.from_pandas(df_pairs)

#maximum length of audio in dset among 'audio_2' and 'audio_1'
max_length = 0
for example in dset:
    if  max(len(example["audio_1"]),len(example["audio_2"])) > max_length:
        max_length =  max(len(example["audio_1"]),len(example["audio_2"]))
def preprocess_function(examples):
    #load audio from path
    inputs = {}
    audio_arrays = [x for x in examples["audio_1"]]
    inputs["audio_1"]  = feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        padding='longest',
    ).pop('input_values')
    audio_arrays = [x for x in examples["audio_2"]]
    inputs["audio_2"] =feature_extractor(
        audio_arrays, 
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_length,
        padding='longest',
    ).pop('input_values')
    return inputs
dset = dset.map(preprocess_function, batched=True)
dset.set_format("torch")
train_dataloader = DataLoader(dset, batch_size=batch_size, shuffle=True, drop_last=True)
y_preds = []
actuals = []
loss_eli5 = contrastiveLossELI5.ContrastiveLoss(batch_size=batch_size)
# actual = torch.ones(len(train_dataloader)*batch_size)

normalize = 0
loss_all = 0
loop_count = 0
for batch in train_dataloader:
    # batch = {k: torch.tensor(v).to(device) for k, v in batch.items()}
    print(f"1-len:{batch['audio_1'].shape}")
    print(f"2-len:{batch['audio_2'].shape}")
    # if batch['audio_1'].shape[1] < 500 or batch['audio_2'].shape[1] < 500:
    #     continue
    #skip if any of the batch["audio_1_length"] or batch["audio_1_length"] is 0
    if 0 in batch["audio_1_length"] or 0 in batch["audio_2_length"]:
        continue

    with torch.no_grad():
        
        outputs_1 = model(batch["audio_1"].to(device)).last_hidden_state


        
        #reduce the number of elements of second dimension in the tensor using elements of audio_1_length
        output_1 = torch.tensor([])
        for i,length in enumerate(batch['audio_1']):
            output_1 = torch.cat((output_1,outputs_1[i][:model._get_feat_extract_output_lengths(len(batch["audio_1"][i])).numpy()].mean(dim=0).unsqueeze(0).cpu()),dim=0)
        # output_1_1 = torch.flip(output_1, [0])
        # output_1 = torch.cat((output_1,output_1_1),dim=0)
        outputs_2 = model(batch["audio_2"].to(device)).last_hidden_state
        output_2 = torch.tensor([])
        for i,length in enumerate(batch['audio_2']):
            output_2 = torch.cat((output_2,outputs_2[i][:model._get_feat_extract_output_lengths(len(batch["audio_2"][i])).numpy()].mean(dim=0).unsqueeze(0).cpu()),dim=0)
        # output_2_1 = torch.flip(output_2, [0])
        # output_2 = torch.cat((output_2,output_2_1),dim=0)
        # output_1 = outputs_1.reshape(-1,output_1.shape[-1])
        # output_2 = outputs_2.reshape(-1,output_2.shape[-1])
        # outputs_1 = outputs_1.mean(dim=1).squeeze().reshape(-1,outputs_1.shape[-1])
        # outputs_2 = outputs_2.mean(dim=1).squeeze().reshape(-1,outputs_2.shape[-1])
        if normalize:
            output_1 = contrastiveLossELI5.F.normalize(output_1, dim=1)
            output_2 = contrastiveLossELI5.F.normalize(output_2, dim=1)
        y_pred = torch.cosine_similarity(output_1,output_2)
        #cosine similarity to 0-1 range
        y_pred = (y_pred+1)/2
        loss_all+=loss_eli5(output_1, output_2)
        y_preds.append(y_pred)
        actual = torch.ones(batch_size)
        # actual = torch.cat((torch.ones(batch_size)),dim=0)#,torch.zeros(batch_size)),dim=0)
        actuals.append(actual)
        loop_count += 1
print(f"loss:{loss_all/loop_count}")
#average precision score
# breakpoint()
#create a torch tensor of shape y_preds having first half the values 1 and next half 0

print(accuracy_score(torch.cat(actuals).cpu().numpy(),torch.cat(y_preds).cpu().numpy().round()))
print("end")
