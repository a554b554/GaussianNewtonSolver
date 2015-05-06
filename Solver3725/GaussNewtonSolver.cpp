#include "hw3_gn.h"

int main() {
	string file = "ellipse.txt";
	ResidualFunction* f = new ellipse_func(file);
	cout << "haha" << endl;
	MySolver ms;
	double X[3] = { 1, 1, 1 };
	ms.solve(f, X);
	int i = 0;
	for (; i < 2; ++i) {
		cout << X[i] << " ";
	}
	cout << X[i] << endl;

	system("pause");
	return 0;
}

int maini() {
	double A = 2.94404;
	double B = 2.30504;
	double C = 1.79783;
	string file = "E:\\ellipse.txt";
	ifstream fin(file.c_str());
	double x, y, z;
	double sum = 0;
	while (fin >> x >> y >> z) {
		double tmp = 1 - (x * x / A / A + y * y / B / B + z * z / C / C);
		sum += tmp * tmp;
	}
	fin.close();

	cout << sum << endl;

	A = 2.94405;
	B = 2.30504;
	C = 1.7982;
	file = "E:\\ellipse.txt";
	fin.open(file.c_str());
	sum = 0;
	while (fin >> x >> y >> z) {
		double tmp = 1 - (x * x / A / A + y * y / B / B + z * z / C / C);
		sum += tmp * tmp;
	}
	fin.close();

	cout << sum << endl;

	system("pause");

	return 0;
}