#ifndef TRANSFORMER_HPP
#define TRANSFORMER_HPP
#include <vector>
#include "InputEmbedding.hpp"
using namespace std;

class Transformer
{
private:
    int seq_len;            //Sequence length size
    int d_model;            //Dimension 
    int num_layers;         //Number of layers in encoder and decoder layer
    int num_heads;          //Number of heads in MultiHeadAttention
    int d_ff;

    InputEmbedding input_embedding;
    
public:
    //Constructor
    Transformer(int seq_len, int d_model, int num_layers, int num_heads, int d_ff);

    //Forward function to build transformer
    vector<vector<float>> build_transformer();
};
#endif //TRANSFORMER_HPP
