import numpy as np

def lu_decomposition(A):
    n = A.shape[0]
    L = np.eye(n)  # Ma trận đơn vị kích thước nxn
    U = A.copy()   # Sao chép ma trận A vào U

    for j in range(n-1):  # Duyệt từ cột 0 đến cột n-2
        if U[j, j] == 0:
            raise ValueError("Ma trận không khả nghịch. Không thể thực hiện phân tích LU.")

        for i in range(j+1, n):  # Duyệt từ hàng j+1 đến hàng n-1
            L[i, j] = U[i, j] / U[j, j]
            U[i, j:] -= L[i, j] * U[j, j:]

    return L, U

if __name__ == "__main__":
    # Tạo một ma trận vuông cấp 5 cụ thể
    A = np.array([
        [0.51867013, 0.25297975, 0.88208041, 0.63406388, 0.65913124],
        [0.29343268, 0.65021791, 0.78143136, 0.32623035, 0.7198189 ],
        [0.14098688, 0.62803225, 0.33284269, 0.11682736, 0.15292955],
        [0.64126897, 0.96450892, 0.21937699, 0.9155413 , 0.80970641],
        [0.19634342, 0.12658216, 0.18050733, 0.46794395, 0.72092556]
    ], dtype=float)

    # Thực hiện phân tích LU
    try:
        L, U = lu_decomposition(A)
        print("Ma trận L (tam giác dưới):")
        print(L)
        print("\nMa trận U (tam giác trên):")
        print(U)
    except ValueError as e:
        print(e)
