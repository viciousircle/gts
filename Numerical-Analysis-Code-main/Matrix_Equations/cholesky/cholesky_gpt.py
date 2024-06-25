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

def solve_cholesky(L, B):
    n = L.shape[0]
    
    # Giải LY = B để tìm Y
    Y = np.zeros_like(B, dtype=float)
    for i in range(n):
        Y[i] = (B[i] - np.dot(L[i, :i], Y[:i])) / L[i, i]
    
    # Giải L^T X = Y để tìm X
    X = np.zeros_like(B, dtype=float)
    for i in range(n-1, -1, -1):
        X[i] = (Y[i] - np.dot(L[i+1:, i], X[i+1:])) / L[i, i]
    
    return X

def save_cholesky_result(L, filename):
    np.savetxt(filename, L, fmt='%.6f', header='Cholesky decomposition result:\n')

if __name__ == "__main__":
    # Tạo một ma trận vuông cấp 7 cụ thể và một vector B
    A = np.array([
        [5, -1, 0, 0, 0, 0, 0],
        [-1, 6, -1, 0, 0, 0, 0],
        [0, -1, 6, -1, 0, 0, 0],
        [0, 0, -1, 6, -1, 0, 0],
        [0, 0, 0, -1, 6, -1, 0],
        [0, 0, 0, 0, -1, 6, -1],
        [0, 0, 0, 0, 0, -1, 5]
    ], dtype=float)

    # Vector B
    B = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)

    try:
        # Thực hiện phân tích Cholesky
        L = cholesky_decomposition(A)
        print("Ma trận L (Phân tích Cholesky):")
        print(L)

        # Lưu kết quả vào file
        save_cholesky_result(L, '/Users/a8888/Documents/doc uni/gts/Numerical-Analysis-Code-main/Matrix_Equations/cholesky/cholesky_gpt_result.txt')
        print("Kết quả đã được lưu vào file '/Users/a8888/Documents/doc uni/gts/Numerical-Analysis-Code-main/Matrix_Equations/cholesky/cholesky_gpt_result.txt'.")

        # Giải hệ phương trình AX = B
        X = solve_cholesky(L, B)
        print("\nNghiệm của hệ phương trình AX = B:")
        print(X)
        
        # Kiểm tra lại bằng AX = B
        print("\nKiểm tra lại: AX = ")
        print(A @ X)
        
    except Exception as e:
        print(e)
