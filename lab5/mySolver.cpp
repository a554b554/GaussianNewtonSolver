//
//  mySolver.cpp
//  lab5
//
//  Created by DarkTango on 4/26/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include "mySolver.h"
void ellipsoid::readDataFromFile(const std::string filename){
    std::fstream fobj;
    fobj.open(filename);
    std::vector<double> tmpvector;
    double tmpdata;
    while (fobj>>tmpdata) {
        tmpvector.push_back(tmpdata);
    }
    int nr = (int)tmpvector.size();
    data = new double[nr];
    for (int i = 0; i < nr; i++) {
        data[i] = tmpvector[i];
        std::cout<<data[i]<<std::endl;
    }
    set(nr, 3);
}

void ellipsoid::set(int nR, int nX){
    _nR = nR;
    _nX = nX;
}

int ellipsoid::nR()const{
    return _nR;
}

int ellipsoid::nX()const{
    return _nX;
}

void ellipsoid::eval(double *R, double *J, double *X){
    for (int i = 0; i < _nR; i++) { //eval R
        double x = data[i*3];
        double y = data[i*3+1];
        double z = data[i*3+2];
        R[i] = (x*x)/(X[0]*X[0]) + (y*y)/(X[1]*X[1]) + (z*z)/(X[2]*X[2]) - 1;
    }
    for (int i = 0; i < _nR; i++) { //eval J
        for (int j = 0; j < _nX; j++) {
            J[i*_nX+j] = -(2*data[i*_nX+j]*data[i*_nX+j])/(X[j]*X[j]*X[j]);
        }
    }
}



ellipsoid::~ellipsoid(){
    delete[] data;
}

void Solver4234::mat2array(int row, int col, cv::Mat &_mat, double *array){
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            array[i*col+j] = _mat.at<double>(i,j);
        }
    }
}

void Solver4234::array2mat(int row, int col, cv::Mat &_mat, double *array){
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            _mat.at<double>(i,j) = array[i*col+j];
        }
    }
}

double dist(double* data, int n){
    double sqsum = 0;
    for (int i = 0; i < n; i++) {
        sqsum += data[i]*data[i];
    }
    return sqrt(sqsum);
}

double matmax(const cv::Mat& _mat){ // to calculate ||R|| and ||detx||
    double max = 0;
    for (int i = 0; i < _mat.rows; i++) {
        double* data = new double[_mat.cols];
        for (int j = 0; j < _mat.cols; j++) {
            data[j] = _mat.at<double>(i,j);
        }
        double t = dist(data,_mat.cols);
        max = t>max?t:max;
    }
    return max;
}

double err(double *R, int n){
    double er = 0;
    for (int i = 0; i < n; i++) {
        er += R[i]*R[i];
    }
    return er;
}

bool checknum(double *X, int n){
    for (int i = 0; i < n; i++) {
        if (X[i] == NAN || X[i] == INFINITY) {
            return false;
        }
    }
    return true;
}

double Solver4234::solve(ResidualFunction *f, double *X, GaussNewtonParams param, GaussNewtonReport *report){
    int itn = 0;
    
    while (1) {
        double *R = new double[f->nR()];
        double *J = new double[f->nR()*f->nX()];
        f->eval(R, J, X);
        std::cout<<"X: "<<X[0]<<" "<<X[1]<<" "<<X[2]<<std::endl;
        
        cv::Mat matJr(f->nR(),f->nX(),CV_64FC1,J);
        cv::Mat matR(f->nR(),1,CV_64FC1,R);
        std::cout<<matR<<std::endl;
        
        //array2mat(f->nR(), f->nX(), matJr, J);
        //array2mat(f->nR(), 1, matR, R);
        cv::Mat detX = -((matJr.t())*(matJr)).inv()*(matJr.t())*(matR);
        std::cout<<detX<<std::endl;
        //write report.
        if (itn >= param.max_iter) {
            if (!checknum(X,f->nX())) {
                report->stop_type = GaussNewtonReport::STOP_NUMERIC_FAILURE;
            }
            else{
                std::cout<<"final error: "<<err(R, f->nR())<<std::endl;
                report->stop_type = GaussNewtonReport::STOP_NO_CONVERGE;
            }
            break;
        }
        
        else if (cv::norm(matJr, cv::NORM_INF) < param.gradient_tolerance) {
            if (!checknum(X,f->nX())) {
                report->stop_type = GaussNewtonReport::STOP_NUMERIC_FAILURE;
            }
            else{
                std::cout<<"final error: "<<err(R, f->nR())<<std::endl;
                report->stop_type = GaussNewtonReport::STOP_GRAD_TOL;
            }
            break;
        }
        else if(cv::norm(matR, cv::NORM_INF) < param.residual_tolerance){
            if (!checknum(X,f->nX())) {
                report->stop_type = GaussNewtonReport::STOP_NUMERIC_FAILURE;
            }
            else{
                std::cout<<"final error: "<<err(R, f->nR())<<std::endl;
                report->stop_type = GaussNewtonReport::STOP_RESIDUAL_TOL;
            }
            break;
        }
        
       // cv::solve(matJr, -matR, detX, cv::DECOMP_SVD);
        std::cout<<"detX: "<<detX<<std::endl;
        std::cout<<"error: "<<err(R, f->nR())<<std::endl;
        int a = 1;
        for (int i = 0; i < f->nR(); i++) {
            X[i] = X[i] + a*detX.at<double>(i,0);
        }
        itn++;
        report->n_iter = itn;
        delete [] R;
        delete [] J;
        
    }
    
    return 0;
}

void Solver4234::printreport(GaussNewtonReport *report){
    if (report == nullptr) {
        std::cout<<"no report!!"<<std::endl;
        return;
    }
    std::cout<<"iterator time: "<<report->n_iter<<std::endl;
    std::cout<<"stop reason: ";
    switch (report->stop_type) {
        case GaussNewtonReport::STOP_GRAD_TOL:
            std::cout<<"Reach Gradient Tolerance."<<std::endl;
            break;
        case GaussNewtonReport::STOP_RESIDUAL_TOL:
            std::cout<<"Reach Residual Tolerance."<<std::endl;
            break;
        case GaussNewtonReport::STOP_NO_CONVERGE:
            std::cout<<"Not Convergence."<<std::endl;
            break;
        case GaussNewtonReport::STOP_NUMERIC_FAILURE:
            std::cout<<"Numeric Failure."<<std::endl;
            break;
        default:
            break;
    }
}











