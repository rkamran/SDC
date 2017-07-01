#include "kalman_filter.h"
#include "tools.h"
#include "iostream"

using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace std;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
    VectorXd y = z - H_ * x_;
    MatrixXd S = H_ * P_ * H_.transpose() + R_;
    MatrixXd K = P_ * H_.transpose() * S.inverse();
    
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    
    //New State Now
    x_ = x_ + (K * y);
    P_ = (I - K * H_) * P_;
    
}

//void KalmanFilter::UpdateEKF(const VectorXd &z) {
//  /**
//  TODO:
//    * update the state by using Extended Kalman Filter equations
//  */
//
//    //1. ***
//    float px = x_(0);
//    float py = x_(1);
//    float vx = x_(2);
//    float vy = x_(3);
//
//    float ro = sqrt(px*px + py*py);
//    float theta = atan2(py, px);
//    float ro_dot = 0;
//    if (ro > 0.0001) ro_dot = (px*vx + py*vy)/ro;
//
//    VectorXd  z_pred_ = VectorXd(4);
//    z_pred_<< ro, theta, ro_dot, 0;
//
//    //END__1. ***
//
//    VectorXd y = z - H_ * z_pred_;
//
//    MatrixXd S = H_ * P_ * H_.transpose() + R_;
//    MatrixXd K = P_ * H_.transpose() * S.inverse();
//
//    long x_size = x_.size();
//    MatrixXd I = MatrixXd::Identity(x_size, x_size);
//
//    //New State Now
//    x_ = x_ + (K * y);
//    P_ = (I - K * H_) * P_;
//
//}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
    float px = x_[0];
    float py = x_[1];
    float vx = x_[2];
    float vy = x_[3];
    
    VectorXd hprime(3);
    float ro = sqrt(pow(px, 2) + pow(py, 2));
    float theta = atan2(py, px);
    float ro_dot = 0;
    if (ro > 0.0001) ro_dot = (px*vx + py*vy)/ro;
    
    hprime << ro, theta, ro_dot;
    
    VectorXd y = z - hprime;
    
    y[1] = atan2(sin(y[1]) ,cos(y[1]));
    
    
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd K = P_ * Ht * S.inverse();
    
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    
    x_ = x_ + (K * y);
    P_ = (I - K * H_) * P_;
    
}
