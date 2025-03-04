#include <iostream>
#include <vector>
#include "HeaderFiles/Transformer.hpp"
using namespace std;

int main(){

    int seq_len;            //Sequence length size
    int d_model;            //Dimension 
    int num_layers;         //Number of layers in encoder and decoder layer
    int num_heads;          //Number of heads in MultiHeadAttention
    int d_ff = 2048;

    cout << "Enter Sequence length: ";
    cin >> seq_len;

    cout << "Enter Embedding dimension(d_model): ";
    cin >> d_model;

    cout << "Enter the number of layers in Encoder and Decoder: ";
    cin >> num_layers;

    cout << "Enter the number of heads: ";
    cin >> num_heads;
        
    Transformer transformer(seq_len, d_model, num_layers, num_heads, d_ff);

    vector<vector<float>> final_output = transformer.build_transformer();

    cout << "Transformer output generated.... \n";

    return 0;
}