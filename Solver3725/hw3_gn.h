#ifndef HW3_GN_34804D67
#define HW3_GN_34804D67

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

struct GaussNewtonParams{
	GaussNewtonParams() :
		exact_line_search(false),
		gradient_tolerance(1e-5),
		residual_tolerance(1e-5),
		max_iter(1000),
		verbose(false)
	{}
	bool exact_line_search; // ʹ�þ�ȷ�����������ǽ�����������
	double gradient_tolerance; // �ݶ���ֵ����ǰ�ݶ�С�������ֵʱֹͣ����
	double residual_tolerance; // ������ֵ����ǰ����С�������ֵʱֹͣ����
	int max_iter; // ����������
	bool verbose; // �Ƿ��ӡÿ����������Ϣ
};

struct GaussNewtonReport {
	enum StopType {
		STOP_GRAD_TOL,       // �ݶȴﵽ��ֵ
		STOP_RESIDUAL_TOL,   // ����ﵽ��ֵ
		STOP_NO_CONVERGE,    // ������
		STOP_NUMERIC_FAILURE // ������ֵ����
	};
	StopType stop_type; // �Ż���ֹ��ԭ��
	double n_iter;      // ��������
};

class ResidualFunction {
public:
	virtual int nR() const = 0;
	virtual int nX() const = 0;
	virtual void eval(double *R, double *J, double *X) = 0;
	virtual double cal(double* X) = 0;
};

class GaussNewtonSolver {
public:
	virtual double solve(
		ResidualFunction *f, // Ŀ�꺯��
		double *X,           // ������Ϊ��ֵ�������Ϊ���
		GaussNewtonParams param = GaussNewtonParams(), // �Ż�����
		GaussNewtonReport *report = nullptr // �Ż��������
		) = 0;
};

class ellipse_func : public ResidualFunction { // ����಻�����κοռ�
	double A = 0, B = 0, C = 0;
	int nr;
	int nx = 3;
	vector<double> datax;
	vector<double> datay;
	vector<double> dataz;
public:
	ellipse_func(int nnr, vector<double> x, vector<double> y, vector<double> z) 
		: nr(nnr), datax(x), datay(y), dataz(z) { }
	ellipse_func(string filename) {
		ifstream fin(filename.c_str());
		double x, y, z;
		while (fin >> x >> y >> z) {
			datax.push_back(x);
			datay.push_back(y);
			dataz.push_back(z);
		}
		nr = datax.size();
		fin.close();
	}
	virtual int nR() const {
		return nr; // ����������ά�ȣ�������753
	}
	virtual int nX() const {
		return nx; // X��ά�ȣ�������3���ֱ����A,B,C
	}
	virtual void eval(double *R, double *J, double *X) { // X�ֱ𱣴����ϴε�ֵ
		for (int i = 0; i < nr; ++i) {
			R[i] = 1 - datax[i] * datax[i] / (X[0] * X[0])
				- datay[i] * datay[i] / (X[1] * X[1]) - dataz[i] * dataz[i] / (X[2] * X[2]);
		}
		for (int i = 0; i < nr; ++i) {
			J[nx * i] = 2 * datax[i] * datax[i] / (X[0] * X[0] * X[0]);
			J[nx * i + 1] = 2 * datay[i] * datay[i] / (X[1] * X[1] * X[1]);
			J[nx * i + 2] = 2 * dataz[i] * dataz[i] / (X[2] * X[2] * X[2]);
		}
		/*for (int i = 0; i < nr; ++i) {
			R[i] = 1 - datax[i] * datax[i] / (X[0] * X[0]) 
				- datay[i] * datay[i] / (X[1] * X[1]) - dataz[i] * dataz[i] / (X[2] * X[2]);
		}*/
	}
	virtual double cal(double* X) {
		double sum = 0;
		for (int i = 0; i < nr; ++i) {
			double tmp = 1 - datax[i] * datax[i] / (X[0] * X[0])
				- datay[i] * datay[i] / (X[1] * X[1]) - dataz[i] * dataz[i] / (X[2] * X[2]);
			sum += tmp * tmp;
		}
		return sum;
	}
};

class MySolver : public GaussNewtonSolver {
public:
	double cal_wp_val(double last, double now, double p, double* g, double* s, double alpha, int N) {
		double sum = 0;
		for (int i = 0; i < N; ++i) {
			sum += alpha * g[i] * s[i];
		}
		return now <= (last + p * sum);
	}

	virtual double GaussNewtonSolver::solve(
		ResidualFunction *f, // Ŀ�꺯��
		double *X,           // ������Ϊ��ֵ�������Ϊ���
		GaussNewtonParams param = GaussNewtonParams(), // �Ż�����
		GaussNewtonReport *report = nullptr // �Ż��������
		) {
		int NR = f->nR();
		int NX = f->nX();
		//cout << "NR = " << NR << ", NX = " << NX << endl;
		double* R = new double[NR];
		double* J = new double[NR * NX];
		double* delta_x = new double[NX];
		double* tmp_x = new double[NX];
		while (true) {
			f->eval(R, J, X);
			//cout << J[NR * NX - 1] << endl;
			
			Mat Jacobi(NR, NX, CV_64FC1, J);
			Mat Residual(NR, 1, CV_64FC1, R);
			Mat JacobiT = Jacobi.t();
			Mat res = (JacobiT * Jacobi).inv() * ((-1) * JacobiT * Residual);

			for (int i = 0; i < NX; ++i) {
				delta_x[i] = res.at<double>(i, 0);
			}
			if (norm(res, NORM_INF) < param.gradient_tolerance
				|| norm(Residual, NORM_INF) < param.residual_tolerance) {
				break;
			}

			double alpha = 1;
			//if (param.exact_line_search) {

			//}
			//else {
			//	cout << "**************" << endl;
			//	double a1 = 0;
			//	double a2 = -1;
			//	double p = 0.01;
			//	double t = 2;

			//	double last_val = f->cal(X);
			//	double sum = 0;
			//	double now_val;
			//	for (int i = 0; i < NR; ++i) {
			//		for (int j = 0; j < NX; ++j) {
			//			sum -= delta_x[j] * Jacobi.at<double>(i, j);
			//		}
			//	}
			//	//cout << "sum = " << last_val << endl;
			//	double last_alpha = alpha;
			//	do {
			//		last_alpha = alpha;
			//		for (int i = 0; i < NX; ++i) {
			//			tmp_x[i] = X[i] + alpha * delta_x[i];
			//		}
			//		now_val = f->cal(tmp_x);

			//		/*cout << "a1 = " << a1 << ", a2 = " << a2 << endl;*/
			//		if (now_val <= (last_val + p * alpha * sum)) {
			//			
			//			if (now_val >= (last_val + (1 - p) * alpha * sum)) {
			//				break;
			//			}
			//			else {
			//				a1 = alpha;
			//				if (a2 == -1) {
			//					//cout << "b" << endl;
			//					alpha *= t;
			//				}
			//				else {

			//					//cout << "a" << endl;
			//					alpha = (a1 + a2) / 2;
			//				}
			//			}
			//		}
			//		else {

			//			//cout << "c" << "a1 = " << a1 << ", a2 = " << a2 << endl;
			//			a2 = alpha;
			//			alpha = (a1 + a2) / 2;
			//			//a2 = alpha;
			//		}
			//		//cout << "alpha = " << alpha << endl;
			//		
			//	} while (a1 != a2 && last_val != now_val && last_alpha != alpha && alpha > 0.001);
			//	//cout << "last = " << last_val << ", now = " << now_val << ", sum = " << sum << endl;
			//	if (alpha < 0.001) {
			//		alpha = 0.001;
			//	}
			//}

			//cout << "alpha = " << alpha << endl;
			for (int i = 0; i < NX; ++i) {
				X[i] += alpha * delta_x[i];
				//cout << "X[" << i << "] = " << X[i] << ", ";
			}
			//cout << delta_x[0] << delta_x[1] << delta_x[2] << endl;
			//cout << endl;
		}
		delete[] R;
		delete[] J;
		delete[] delta_x;
		delete[] tmp_x;
		return 0;
	}

};

#endif /* HW3_GN_34804D67 */
