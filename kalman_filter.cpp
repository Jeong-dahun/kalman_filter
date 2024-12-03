#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

class KalmanFilter {
public:
    KalmanFilter(int state_dim, int meas_dim) {
        x = VectorXd::Zero(state_dim);
        P = MatrixXd::Identity(state_dim, state_dim);
        F = MatrixXd::Identity(state_dim, state_dim);
        H = MatrixXd::Zero(meas_dim, state_dim);
        R = MatrixXd::Identity(meas_dim, meas_dim);
        Q = MatrixXd::Identity(state_dim, state_dim);
    }

    void predict() {
        x = F * x;
        P = F * P * F.transpose() + Q;
    }

    void update(const VectorXd &z) {
        VectorXd y = z - H * x;
        MatrixXd S = H * P * H.transpose() + R;
        MatrixXd K = P * H.transpose() * S.inverse();
        x = x + K * y;
        P = (MatrixXd::Identity(x.size(), x.size()) - K * H) * P;
    }

    VectorXd getState() {
        return x;
    }

    void setTransitionMatrix(const MatrixXd &F_in) {
        F = F_in;
    }

    void setMeasurementMatrix(const MatrixXd &H_in) {
        H = H_in;
    }

    void setProcessNoise(const MatrixXd &Q_in) {
        Q = Q_in;
    }

    void setMeasurementNoise(const MatrixXd &R_in) {
        R = R_in;
    }

private:
    VectorXd x;
    MatrixXd P, F, H, R, Q;
};

int main() {
    int state_dim = 6; // 상태 벡터 크기: [x, y, vx, vy, ax, ay]
    int meas_dim = 3;  // 관측 벡터 크기: [x, y, vx]

    KalmanFilter kf(state_dim, meas_dim);

    // 상태 전이 행렬 F
    MatrixXd F(state_dim, state_dim);
    F << 1, 0, 1, 0, 0.5, 0,
         0, 1, 0, 1, 0, 0.5,
         0, 0, 1, 0, 1, 0,
         0, 0, 0, 1, 0, 1,
         0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1;
    kf.setTransitionMatrix(F);

    // 관측 행렬 H
    MatrixXd H(meas_dim, state_dim);
    H << 1, 0, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0,
         0, 0, 1, 0, 0, 0;
    kf.setMeasurementMatrix(H);

    // 프로세스 노이즈 Q
    MatrixXd Q = MatrixXd::Identity(state_dim, state_dim) * 0.1;
    kf.setProcessNoise(Q);

    // 관측 노이즈 R
    MatrixXd R = MatrixXd::Identity(meas_dim, meas_dim) * 0.1;
    kf.setMeasurementNoise(R);

    // 초기 관측값 (위치와 속도만 측정 가능)
    VectorXd z(meas_dim);
    z << 2, 3, 1; // x = 2, y = 3, vx = 1

    // 칼만 필터 실행
    kf.predict();
    kf.update(z);

    // 추정된 상태 출력
    cout << "Estimated state:\n" << kf.getState() << endl;

    return 0;
}