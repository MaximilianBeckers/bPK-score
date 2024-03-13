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

import torch.nn as nn
import torch

class Predictor_onlyGlobalFeats_wdropout(nn.Module):
    
    def __init__(self, global_feats=0, n_tasks=1, predictor_hidden_feats=128, predictor_dropout=0., num_layers=1):
        super(Predictor_onlyGlobalFeats_wdropout, self).__init__();
        
        if num_layers == 1:
            mlp = [nn.Dropout(predictor_dropout), nn.Linear(global_feats, n_tasks-1)];
        else:
            mlp = [nn.Dropout(predictor_dropout), nn.Linear( global_feats, predictor_hidden_feats), nn.BatchNorm1d(predictor_hidden_feats), nn.ReLU()];
            for _ in range(num_layers - 2):
                mlp.extend([nn.Dropout(predictor_dropout), nn.Linear(predictor_hidden_feats, predictor_hidden_feats), nn.BatchNorm1d(predictor_hidden_feats), nn.ReLU()]);
            mlp.extend([nn.Linear(predictor_hidden_feats, n_tasks-1)]);
        
        self.predict = nn.Sequential(*mlp);

    def forward(self, global_feats):
                        
            logits = self.predict(global_feats);
            probas = torch.sigmoid(logits);
            
            return logits, probas;