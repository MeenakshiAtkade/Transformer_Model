#include "../HeaderFiles/Transformer.hpp"
#include "../HeaderFiles/InputEmbedding.hpp"
#include "../HeaderFiles/PositionalEncoding.hpp"
#include "../HeaderFiles/Encoder.hpp"
#include <vector>
#include <iostream>
using namespace std;

Transformer::Transformer(int seq_len, int d_model, int num_layers, int num_heads, int d_ff):seq_len(seq_len), d_model(d_model), num_layers(num_layers), num_heads(num_heads), d_ff(d_ff)
{
    cout << "Transformer initialized with the following parameters:" << endl;
    cout << "Sequence length: " << seq_len << ", Embedding dimension: " << d_model << endl;
    cout << "Number of layers: " << num_layers << ", Number of heads: " << num_heads << endl;
}

vector<vector<float>> Transformer::build_transformer()
{
    //Call Input Embedding
    InputEmbedding input_embedding(seq_len, d_model);
    vector<vector<float>> embedded_matrix = input_embedding.forward();

    //Call Positional Encoding
    PositionalEncoding positional_encoding(seq_len, d_model);
    vector<vector<float>> positional_encoded_matrix = positional_encoding.forward(embedded_matrix);

    //Call Encoder
    Encoder encoder(seq_len, d_model, num_layers, num_heads, d_ff);
    vector<vector<float>> encoder_resultant_matrix = encoder.forward(positional_encoded_matrix);

    return encoder_resultant_matrix;
}
