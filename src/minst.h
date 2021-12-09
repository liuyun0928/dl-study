#pragma once
#include <vector>

namespace liuyun {
    class Minst
    {
    public:
        Minst(std::vector<int> sizes);
        ~Minst();
        double feedForword();

        /**
         * @brief SGD: Stochastic Gradient Descent
         * 
         * @param train_data 
         * @param epochs 
         * @param mini_batch_size 
         */
        void SGD(std::vector<double> train_data, int epochs, int mini_batch_size);

        /**
         * @brief update mini batch
         * 
         * @param mini_batch 
         */
        void UpdateMiniBatch(std::vector<int> mini_batch);

        /**
         * @brief return
         * 
         * @param x 
         * @param y 
         * @return std::vector<int> 
         */
        std::vector<int> backprop(int x, int y);

        /**
         * @brief return the number of test inputs for which the neural networks outputs the correct result.
         * Note that the nn's output is assumed to be the index of whichever neruon in the final lay has the
         * highest activation.
         * 
         * @param test_data 
         * @return bool 
         */
        bool evaluate(std::vector<int> test_data);

        /**
         * @brief return the vector of partial derivatives.
         * 
         * @param output_activations 
         * @param y 
         * @return std::vector<double> 
         */
        std::vector<double> costDerivative(std::vector<double> output_activations, int y);

        /**
         * @brief sigmod function 
         * 
         * @param z 
         * @return double 
         */
        double sigmoid(double z);

        /**
         * @brief derivative of the sigmod function.
         * 
         * @param z 
         * @return double 
         */
        double sigmoidPrime(double z);
    private:
        int n_layers;
        std::vector<int> sizes;
        std::vector<double> biases;
        std::vector<double> weights;
    };
}