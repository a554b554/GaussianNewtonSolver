//
//  mySolver.h
//  lab5
//
//  Created by DarkTango on 4/26/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#ifndef __lab5__mySolver__
#define __lab5__mySolver__

#include <opencv2/core/core.hpp>
#include <string>
#include <fstream>
#include <vector>
#include "GaussianNewtonSolver.h"
#include <iostream>
class Solver4234: public GaussNewtonSolver{
public:
    double solve(
                 ResidualFunction *f, // 目标函数
                 double *X,           // 输入作为初值，输出作为结果
                 GaussNewtonParams param = GaussNewtonParams(), // 优化参数
                 GaussNewtonReport *report = nullptr // 优化结果报告
    );
    void printreport(GaussNewtonReport *report);
private:
    void mat2array(int row, int col, cv::Mat& _mat, double* array);
    void array2mat(int row, int col, cv::Mat& _mat, double* array);
    
};

class ellipsoid: public ResidualFunction{
public:
    int nR() const;
    int nX() const;
    void eval(double *R, double *J, double *X);
    void readDataFromFile(const std::string filename);
    ~ellipsoid();
private:
    void set(int nR, int nX);
    int _nR;
    int _nX;
    double* data;
};


#endif /* defined(__lab5__mySolver__) */
