#ifndef POSITIONAL_ENCODING_HPP
#define POSITIONAL_ENCODING_HPP
#include <vector>
using namespace std;

class PositionalEncoding{
private:
    int seq_len;
    int d_model;

public:
    //Constructor
    PositionalEncoding(int seq_len, int d_model);

    //Function to generate positional encoding
    vector<vector<float>> forward(const vector<vector<float>>& input_embedding);
    
};
#endif //POSITIONAL_ENCODING_HPP