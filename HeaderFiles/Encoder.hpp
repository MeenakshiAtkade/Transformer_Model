#ifndef ENCODER_HPP
#define ENCODER_HPP
#include <vector>
using namespace std;

class Encoder
{
private:
    int seq_len;
    int d_model;
    int num_layers;
    int num_heads;
    int d_ff;

public:
    //Constructor
    Encoder(int seq_len, int d_model, int num_layers, int num_heads, int d_ff);

    //Forward function
    vector<vector<float>> forward(const vector<vector<float>>& embedded_matrix);

};
#endif //ENCODER_HPP

