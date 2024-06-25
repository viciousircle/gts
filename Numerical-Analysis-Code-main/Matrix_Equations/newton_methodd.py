import numpy as np
import pandas as pd

def newton_raphson(F, J, x0, tol=1e-6, max_iter=100):
    x = x0
    x_history = [x]  # List to store the history of x values
    for i in range(max_iter):
        Fx = F(x)
        Jx = J(x)
        delta_x = np.linalg.solve(Jx, -Fx)
        x = x + delta_x
        x_history.append(x.copy())  # Copy x to avoid overwriting in the list
        if np.linalg.norm(delta_x, ord=2) < tol:
            return x_history
    raise ValueError("Phương pháp Newton-Raphson không hội tụ sau {} lần lặp".format(max_iter))

def F_vi_du(x):
    F1 = x[0]**2 + x[1] - 37
    F2 = x[0] - x[1]**2 - 5
    return np.array([F1, F2])

def J_vi_du(x):
    J11 = 2*x[0]
    J12 = 1
    J21 = 1
    J22 = -2*x[1]
    return np.array([[J11, J12], [J21, J22]])

def main():
    x0 = np.array([0.0, 0.0])  # Giá trị ban đầu

    try:
        x_history = newton_raphson(F_vi_du, J_vi_du, x0)
    except ValueError as e:
        print(e)
        return

    # Tạo DataFrame từ x_history
    n_iterations = len(x_history)
    df = pd.DataFrame(x_history, columns=['x1', 'x2'])
    df.index.name = 'Lần lặp'

    # In ra bảng các lần lặp và nghiệm cuối cùng
    print("Bảng các lần lặp:")
    pd.options.display.float_format = '{:.6f}'.format  # Định dạng số thập phân
    pd.set_option('display.width', None)  # Tự động điều chỉnh độ rộng của cột
    print(df)

    # In ra nghiệm cuối cùng
    print("\nNghiệm tìm được:")
    print(x_history[-1])

if __name__ == "__main__":
    main()
