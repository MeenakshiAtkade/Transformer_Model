#ifndef INPUT_EMBEDDING_HPP
#define INPUT_EMBEDDING_HPP
#include <vector>
using namespace std;

class InputEmbedding
{
public:
    int seq_len;
    int d_model;

    //Constructor
    InputEmbedding();
    InputEmbedding(int seq_len, int d_model);
    
    //Forward function to create input embedding
    vector<vector<float>> forward();
};
#endif //INPUT_EMBEDDING_HPP

