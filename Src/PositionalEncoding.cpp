#include "../HeaderFiles/PositionalEncoding.hpp"
#include<iostream>
#include <cmath>

using namespace std;

PositionalEncoding::PositionalEncoding(int seq_len, int d_model) : seq_len(seq_len), d_model(d_model)
{
    cout << "Positional Encoding constructor is called......." << endl;
}

//Forward function to generate positional encoding
vector<vector<float>> PositionalEncoding::forward(const vector<vector<float>> &input_embedding)
{
    vector<vector<float>> positional_encoding(seq_len, vector<float>(d_model, 0.0));

    for(int pos = 0; pos < seq_len; pos++){
        for(int i = 0; i < d_model; i += 2){
            float angle = pos / pow(10000, (2.0 * i) / d_model);
            positional_encoding[pos][i] = sin(angle);
            if(i + 1 < d_model){
                positional_encoding[pos][i + 1] = cos(angle);
            }
        }
    }

    //Print positional encoding matrix
    cout << "Positional Encoding matrix: " << endl;
    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j < d_model; j++){
            cout << positional_encoding[i][j] << " ";
        }
        cout << endl;
    }

    //Add positional encoding and input embedding matrix
    vector<vector<float>> final_embedding(seq_len, vector<float>(d_model, 0.0));
    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j < d_model; j++){
            final_embedding[i][j] = input_embedding[i][j] + positional_encoding[i][j];
        }
    }

    //Print final embedded matrix
    cout << "Final Embedded matrix: " << endl;
    for(int i = 0; i < seq_len; i++){
        for(int j = 0; j < d_model; j++){
            cout << final_embedding[i][j] << " ";
        }
        cout << endl;
    }

    return final_embedding;
}