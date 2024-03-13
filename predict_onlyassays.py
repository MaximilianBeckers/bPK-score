#  Copyright (c) 2023, Novartis Institutes for BioMedical Research Inc.
#  All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: 
#
#     * Redistributions of source code must retain the above copyright 
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following 
#       disclaimer in the documentation and/or other materials provided 
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc. 
#       nor the names of its contributors may be used to endorse or promote 
#       products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import importlib
import torch
import numpy as np
from torch.utils.data import DataLoader

device='cpu';
    
#****************************************************************************
#****************** some pre-defined lists of features **********************
#****************************************************************************
#IMPORTANT: THESE HAVE TO BE SET MANUALLY
features_to_be_scaled= TO_BE_SET
global_features = TO_BE_SET


#***********************************************************************************
def get_data( global_features, labels ):

    bPKs= [];
    global_feat_list = [];

    for i, smi in enumerate(labels):
        
        bPKs.append(int(labels[i]));
        global_feat_list.append(torch.from_numpy(global_features[i,:])); 
        
    return  global_feat_list, bPKs;
    
#***********************************************************************************   
def collate(sample):
    global_feats, labels = map(list,zip(*sample));
    global_feats = torch.stack([torch.tensor(tmp) for tmp in global_feats]);

    return  global_feats, torch.tensor(labels);

#**********************************************************************************
def scale_features(df):
    from sklearn.preprocessing import StandardScaler
    import pickle
    scalerfile = "scaler_onlyassays.sav";
    final_train_scaler = pickle.load(open(scalerfile, 'rb'));

    df[features_to_be_scaled] = final_train_scaler.transform(df[features_to_be_scaled]);

    return df;

#***********************************************************************************
def predict(df, model_path, scale=True, out_rank=1):
    
    df = df.copy();
    
    print("Scaling the features ...");
    if scale:
        df = scale_features(df);

    import sys
    sys.path.insert(1, model_path);
    mod = importlib.import_module('Predictor_onlyGlobalFeats_wdropout');
    importlib.reload(mod);
    Predictor_onlyGlobalFeats_wdropout = getattr(mod, 'Predictor_onlyGlobalFeats_wdropout');

    num_models = 10;
    model_list = [];

    print("Read the models from " + str(model_path));
    for i in range(10):
        tmp_model = Predictor_onlyGlobalFeats_wdropout( global_feats=len(global_features), num_layers=7,
                                           n_tasks=5, predictor_hidden_feats=256);

        tmp_model.load_state_dict(torch.load(model_path + "/weights_" + str(i) + ".pth", map_location=torch.device('cpu')));
        tmp_model.eval();
        model_list.append(tmp_model);

    print("Make graphs ...");
    global_feat_list, bPK = get_data(df[global_features].to_numpy(dtype=np.float32), 
                                                [1]*df.shape[0]);
    
    test_data = list(zip( global_feat_list, bPK)).copy();
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, collate_fn=collate, drop_last=False);

    print("Make predictions ...");
    preds = [];
    model_list_device = [tmp_model.to(device) for tmp_model in model_list];

    for i, ( global_feats, labels) in enumerate(test_loader):            
        labels = labels.to(device);
        global_feats = global_feats.to(device);
        
        probs = np.zeros(len(labels));
        for tmp_model in model_list_device:
            logits, probas = tmp_model( global_feats);
            tmp_probs = probas.detach().to('cpu').numpy();
            if out_rank==-1:
                # take the sum over the predicted probabilities
                probs = probs + (1.0/num_models)* (tmp_probs[:,0] + tmp_probs[:,1] + tmp_probs[:,2] + tmp_probs[:,3]); 
            else:
                probs = probs + (1.0/num_models)* tmp_probs[:,out_rank];

        preds = preds + list(probs);
        
    return preds;
  
    