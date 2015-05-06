//
//  main.cpp
//  lab5
//
//  Created by DarkTango on 4/25/15.
//  Copyright (c) 2015 DarkTango. All rights reserved.
//

#include <iostream>
#include "mySolver.h"
using namespace std;
int main() {
    Solver4234 *solver = new Solver4234();
    ellipsoid *f = new ellipsoid();
    f->readDataFromFile("ellipse753.txt");
    GaussNewtonReport *report = new GaussNewtonReport();
    double *X = new double[3];
    X[0] = 1;
    X[1] = 1;
    X[2] = 1;
    GaussNewtonParams para;
    solver->solve(f, X, GaussNewtonParams(), report);
    solver->printreport(report);
    cout<<X[0]<<" "<<X[1]<<" "<<X[2]<<endl;
    return 0;
}
