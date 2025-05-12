import torch
import torch.nn as nn

class IMDbCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pre_trained_embeddings=None,
                 kernel_width=3, conv_out_channels=8, pool_size=8,
                 dense_in_features=128, dense_out_features=1, pad_idx=None):
        super(IMDbCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if pre_trained_embeddings is not None:      
            self.embedding.weight.data.copy_(pre_trained_embeddings)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, 
                                out_channels=conv_out_channels, 
                                kernel_size=kernel_width,
                                padding=0) 

        self.relu = nn.ReLU()
        self.maxpool1d = nn.MaxPool1d(kernel_size=pool_size, 
                                      stride=pool_size) 
        self.fc = nn.Linear(dense_in_features, dense_out_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x) 
        embedded_permuted = embedded.permute(0, 2, 1) 
        conved = self.conv1d(embedded_permuted)
        activated = self.relu(conved)
        pooled = self.maxpool1d(activated)
        flattened = torch.flatten(pooled, 1) 
        dense_out = self.fc(flattened)
        output = self.sigmoid(dense_out)
        
        return output

if __name__ == '__main__':  
    VOCAB_SIZE_TEST = 1000
    EMBEDDING_DIM_TEST = 50
    SEQ_LEN_TEST = 130 
    BATCH_SIZE_TEST = 32
    DENSE_IN_FEATURES_TEST = 128

    
    test_model = IMDbCNN(VOCAB_SIZE_TEST, EMBEDDING_DIM_TEST, 
                         kernel_width=3, conv_out_channels=8, pool_size=8,
                         dense_in_features=DENSE_IN_FEATURES_TEST)
    
    dummy_input = torch.randint(0, VOCAB_SIZE_TEST, (BATCH_SIZE_TEST, SEQ_LEN_TEST)) 
    output = test_model(dummy_input)
    print("Test model defined successfully.")
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape) 
    
    
    total_params = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")