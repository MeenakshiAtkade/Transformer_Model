#include "../HeaderFiles/MultiHeadAttention.hpp"
#include <vector>
#include <iostream>
using namespace std;

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads)
{
    cout << "MultiHeaded Attention constructor is called......";
    cout << "d_model: " << d_model << "  "  << "No. of heads: " << num_heads;
}

vector<vector<float>> MultiHeadAttention::forward(const vector<vector<float>> &embedded_matrix)
{
    return vector<vector<float>>();
}
