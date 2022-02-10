//
// Created by gavin on 2022/2/10.
//

#include "Net.h"
#include "Utils.h"
#include <random>

Net::Net() {
    std::mt19937 rd;
    rd.seed(std::random_device()());

    std::uniform_real_distribution<double> distribution(-1, 1);

    // 初始化输入层
    for (size_t i = 0; i < Config::INNODE; ++i) {
        inputLayer[i] = new Node(Config::HIDENODE);
        // 输入层的神经元节点不需要偏置值
        // inputLayer[i]->bias = distribution(rd);
        for (size_t j = 0; j < Config::HIDENODE; ++j) {
            // 初始化输入层第 i 个神经元到隐藏层第 j 个神经元的权重
            inputLayer[i]->weight[j] = distribution(rd);
            // 初始化输入层第 i 个神经元到隐藏层第 j 个神经元的权重修正值
            inputLayer[i]->weight_delta[j] = 0.f;
        }
    }

    // 初始化隐藏层
    for (size_t j = 0; j < Config::HIDENODE; ++j) {
        hideLayer[j] = new Node(Config::OUTNODE);
        // 初始化隐藏层神经元系节点偏置值
        hideLayer[j]->bias = distribution(rd);
        // 初始化隐藏层神经元系节点偏置值修正值
        hideLayer[j]->bias_delta = 0.f;
        for (size_t k = 0; k < Config::OUTNODE; ++k) {
            // 初始化隐藏层第 j 个神经元到输出层第 k 个神经元的权重
            hideLayer[j]->weight[k] = distribution(rd);
            // 初始化隐藏层第 j 个神经元到输出层第 k 个神经元的权重修正值
            hideLayer[j]->weight_delta[k] = 0.f;
        }
    }

    // 初始化输出层
    for (size_t k = 0; k < Config::OUTNODE; ++k) {
        outputLayer[k] = new Node(0);
        // 初始化输出层神经元系节点偏置值
        outputLayer[k]->bias = distribution(rd);
        // 初始化输出层神经元系节点偏置值偏置值
        outputLayer[k]->bias_delta = 0.f;
    }
}

void Net::grad_zero() {

    // 清零输入层所有节点的 weight_delta
    for (auto &node_input: inputLayer) {
        node_input->weight_delta.assign(node_input->weight_delta.size(), 0.f);
    }

    // 清零隐藏层所有节点的 bias_delta 和 weight_delta
    for (auto &node_hide: hideLayer) {
        node_hide->bias_delta = 0.f;
        node_hide->weight_delta.assign(node_hide->weight_delta.size(), 0.f);
    }

    // 清零输出层所有节点的 bias_delta
    for (auto &node_output: outputLayer) {
        node_output->bias_delta = 0.f;
    }
}

void Net::forward() {

    // 输入层向隐藏层传播
    for (size_t j = 0; j < Config::HIDENODE; ++j) {
        // 计算第 j 个隐藏层节点的值
        double sum = 0;
        for (size_t i = 0; i < Config::INNODE; ++i) {
            // 第 i 个输入层节点对第 j 个隐藏层节点的贡献
            sum += inputLayer[i]->value * inputLayer[i]->weight[j];
        }
        sum -= hideLayer[j]->bias;

        hideLayer[j]->value = Utils::sigmoid(sum);
    }

    // 隐藏层向输出层传播
    for (size_t k = 0; k < Config::OUTNODE; ++k) {
        // 计算第 k 个输出层节点的值
        double sum = 0;
        for (size_t j = 0; j < Config::HIDENODE; ++j) {
            // 第 j 个隐藏层节点对第 k 个输出层节点的恭喜
            sum += hideLayer[j]->value * hideLayer[j]->weight[k];
        }
        sum -= outputLayer[k]->bias;

        outputLayer[k]->value = Utils::sigmoid(sum);
    }
}

double Net::CalculateLoss(const vector<double> &out) {
    double loss = 0.f;

    for (size_t k = 0; k < Config::OUTNODE; ++k) {
        double tmp = std::fabs(outputLayer[k]->value - out[k]);
        loss += tmp * tmp / 2;
    }

    return loss;
}

Node::Node(int size) {
    weight.resize(size);
}
