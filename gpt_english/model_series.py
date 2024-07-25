import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from einops import rearrange
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


class GPT2TS(nn.Module):
    def __init__(self, configs, device, gloss_features, boolean_features, inflection_features):
        super(GPT2TS, self).__init__()
        self.vocab_size = configs.vocab_size
        self.device = device
        self.gloss_features = gloss_features
        self.boolean_features = boolean_features
        self.inflection_features = inflection_features

        # GPT-2 setup
        if configs.is_gpt:
            gpt2_config = GPT2Config(vocab_size=self.vocab_size)
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('gpt2', config=gpt2_config)
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(gpt2_config)
            
            self.gpt2.config.n_layer = configs.gpt_layers
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            
            print("gpt2 = {}".format(self.gpt2))

            if configs.freeze:
                for param in self.gpt2.parameters():
                    param.requires_grad = False
                
                # Optionally unfreeze some layers (e.g., the last layer)
                for param in self.gpt2.h[-1].parameters():
                    param.requires_grad = True

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=configs.d_model, 
            nhead=configs.transformer_heads, 
            dim_feedforward=configs.d_model * 4,
            dropout=configs.dropout
        )
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=configs.transformer_layers
        )

        self.frame_lstm = nn.LSTM(
            configs.d_model, 
            configs.lstm_hidden_size, 
            num_layers=configs.lstm_layers, 
            batch_first=True,
            dropout=configs.dropout if configs.lstm_layers > 1 else 0
        )

        # Output layers for various features
        self.out_layer_gloss = nn.ModuleDict({
            gloss: nn.Linear(configs.d_model, configs.vocab_size) for gloss in gloss_features
        })
        
        self.out_layer_frame = nn.Linear(configs.lstm_hidden_size, 1)
        
        self.boolean_features_output = nn.ModuleDict({
            feature: nn.Linear(configs.lstm_hidden_size, 5) for feature in boolean_features
        })

        self.inflection_feature_outputs = nn.ModuleDict({
            feature: nn.Linear(configs.lstm_hidden_size, 1) for feature in inflection_features
        })
        
        self.dropout = nn.Dropout(configs.dropout)

        # Move all layers to the specified device
        self.to(device)

    def forward(self, text_tokens):
        gpt_output = self.gpt2(text_tokens).last_hidden_state
        gpt_output = self.dropout(gpt_output)

        transformer_output = self.transformer(gpt_output.transpose(0, 1)).transpose(0, 1)
        transformer_output = self.dropout(transformer_output)
        
        # Generate outputs for gloss features
        gloss_logits = {gloss: self.out_layer_gloss[gloss](transformer_output) for gloss in self.gloss_features}

        # Process through LSTM for frame and other features
        frame_lstm_out, _ = self.frame_lstm(transformer_output)
        frame_lstm_out = self.dropout(frame_lstm_out)
        
        framestart_values = self.out_layer_frame(frame_lstm_out)

        boolean_outputs = {feature: self.boolean_features_output[feature](frame_lstm_out) 
                           for feature in self.boolean_features}

        inflection_feature_outputs = {feature: self.inflection_feature_outputs[feature](frame_lstm_out).squeeze(-1) 
                                      for feature in self.inflection_features}

        return gloss_logits, framestart_values.squeeze(-1), boolean_outputs, inflection_feature_outputs        