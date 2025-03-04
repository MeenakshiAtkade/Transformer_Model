#include "../HeaderFiles/InputEmbedding.hpp"
#include <iostream>
#include <vector>
using namespace std;

InputEmbedding::InputEmbedding()
{
}

InputEmbedding::InputEmbedding(int seq_len, int d_model) 
{
    this->seq_len = seq_len;
    this->d_model = d_model;
    cout << "InputEmbedding initialized initialized with the following parameters:" << endl;
    cout << "Sequence length: " << seq_len << ", Embedding dimension: " << d_model << endl;
}

vector<vector<float>> InputEmbedding::forward()
{
    //Create input sequence
    vector<vector<float>> input_embedding(seq_len, vector<float>(d_model));
    for(int i = 0; i < seq_len; ++i){
        for(int j = 0; j < d_model; ++j){
            input_embedding[i][j] = (float)0.0;
        }
    }

    //Print the input embedding
    cout << "Input Eembedding: " << endl;
    for(int i = 0; i < seq_len; ++i){
        for(int j = 0; j < d_model; ++j){
            cout << input_embedding[i][j] << " ";
        }
        cout << endl;
    }
    
    return input_embedding;
}
