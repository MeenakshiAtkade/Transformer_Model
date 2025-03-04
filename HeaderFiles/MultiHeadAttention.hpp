#ifndef MULTIHEADATTENTION_HPP
#define MULTIHEADATTENTION_HPP
#include <vector>
using namespace std;

class MultiHeadAttention
{
private:
    int d_model;
    int num_heads;
    int key;                                    //Dimension of key
    int query;                                  //Dimension of query
    int value;                                  //Dimension of value

    //Weight matrix for each head
    vector<vector<vector<float>>> W_query;      //Query weight matrix
    vector<vector<vector<float>>> W_key;        //Key weight matrix
    vector<vector<vector<float>>> W_value;      //Value weight matrix
    vector<vector<vector<float>>> W_output;     //Output weight matrix

public:
    //Constructor
    MultiHeadAttention(int d_model, int num_heads);

    //Forward function
    vector<vector<float>> forward(const vector<vector<float>>& embedded_matrix);
};
#endif //MULTIHEADATTENTION_HPP




