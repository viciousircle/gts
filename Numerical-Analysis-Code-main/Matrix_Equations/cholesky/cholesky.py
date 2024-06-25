import numpy as np

def cholesky_decomposition(A):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)

    for j in range(n):
        for i in range(j, n):
            if i == j:
                # Tính L[i, i]
                L[i, i] = np.sqrt(A[i, i] - np.sum(L[i, :i]**2))
            else:
                # Tính L[i, j]
                L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]

    return L

def save_cholesky_result(L, filename):
    np.savetxt(filename, L, fmt='%.6f', header='Cholesky decomposition result:\n')

if __name__ == "__main__":
    # Tạo một ma trận vuông cấp 8 cụ thể
    A = np.array([
    [10, 1, 4, 2, 3],
    [1, 12, -3, 0, -5],
    [4, -3, 15, 6, 2],
    [2, 0, 6, 8, 1],
    [3, -5, 2, 1, 10]
    ], dtype=float)
    try:
        # Thực hiện phân tích Cholesky
        L = cholesky_decomposition(A)
        print("Ma trận L (Phân tích Cholesky):")
        print(L)

        # Lưu kết quả vào file
        filename = "/Users/a8888/Documents/doc uni/gts/Numerical-Analysis-Code-main/Matrix_Equations/cholesky/cholesky_result.txt"
        save_cholesky_result(L, filename)
        print(f"Kết quả đã được lưu vào file '{filename}'.")

        # Kiểm tra lại bằng L * L^T
        print("\nKiểm tra lại: L * L^T = ")
        print(L @ L.T)
    except Exception as e:
        print(e)
