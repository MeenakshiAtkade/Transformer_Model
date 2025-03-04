#include "../HeaderFiles/Encoder.hpp"
#include "../HeaderFiles/MultiHeadAttention.hpp"
#include <vector>
#include <iostream>
using namespace std;

Encoder::Encoder(int seq_len, int d_model, int num_layers, int num_heads, int d_ff): seq_len(seq_len), d_model(d_model), num_layers(num_layers), num_heads(num_heads), d_ff(d_ff)
{
    cout << "Encoder constructor is called.........." << endl;
}

vector<vector<float>> Encoder::forward(const vector<vector<float>> &embedded_matrix)
{
    for(int i = 0; i < num_layers; i++){
        //Call MultiHeadAttention
        MultiHeadAttention multiheaded_attention(d_model, num_heads);
        vector<vector<float>> attention_output = multiheaded_attention.forward(embedded_matrix);

        //Call LayerNormalization


        //Call FeedForward

        
        //Call LayerNormalization
    }

    return vector<vector<float>>();
}
