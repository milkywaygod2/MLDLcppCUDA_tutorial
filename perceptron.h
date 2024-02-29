#pragma once
#include "_pch.h"

#define DATA_NUMS		4
#define WEIGHT_NUMS		3

class Cperceptron {
	friend int main();
private:
	int epoch = 10;
	float e = 0.1;
	float Xij[DATA_NUMS][WEIGHT_NUMS] = {{1,0,0},{1,0,1},{1,0,1},{1,1,1}};
	float* pXij = Xij[0];
	float z = 0;
	float Ti[DATA_NUMS] = {0,0,0,1}; //논리곱
	//float t[DATA_NUMS] = {0,1,1,1}; //논리합
	float Wj[WEIGHT_NUMS] = {0,0,0};

public:
	float isActive(float _xw) { return _xw > 0 ? 1 : 0; }
	float dotSigma(float* _Xj, float* _Wj, int _lenWj) {
		float grossSum = 0;
		for(int k = 0; k < _lenWj; k++) { grossSum += _Xj[k] * _Wj[k]; } //accumulate _xw
		return grossSum;
	}
	float forward(float* _Xj, float* _Wj, int _lenWj) {
		float u = dotSigma(_Xj, _Wj, _lenWj);
		return isActive(u);
	}
	void train(float* _Xj, float* _Wj, float t, float& z, float _e, int _lenWj) {
		z = forward(_Xj, _Wj, _lenWj);
		for(int j = 0; j < _lenWj; j++) { _Wj[j] += (t - z) * _Xj[j] * e; } //★
	}
	int perceptron() { //float** _Xij, float* _Wj, float* _Ti, float _e, int _lenXi, int _lenWj
		for(int h = 0; h < epoch; h++) {
			cout << "epoch : " << h << " ";
			for(int i = 0; i < DATA_NUMS; i++) {
				train(Xij[i], Wj, Ti[i], z, e, WEIGHT_NUMS);
			}
			for(int i = 0; i < WEIGHT_NUMS; i++) {
				cout << "w" << i << " : " << Wj[i] << " ";
			}
			cout << endl;
		}
		for(int i = 0; i < DATA_NUMS; i++) {
			cout << forward(Xij[i], Wj, WEIGHT_NUMS) << " ";
		}
		cout << endl;
		return 0;
	}
public:
	Cperceptron(){}
	~Cperceptron(){}
};