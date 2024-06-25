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

def solve_lu(L, U, B):
    n = L.shape[0]
    
    # Giải LY = B để tìm Y
    Y = np.zeros_like(B, dtype=float)
    for i in range(n):
        Y[i] = B[i] - np.dot(L[i, :i], Y[:i])
    
    # Giải UX = Y để tìm X
    X = np.zeros_like(B, dtype=float)
    for i in range(n-1, -1, -1):
        X[i] = (Y[i] - np.dot(U[i, i+1:], X[i+1:])) / U[i, i]

    return X

def save_solution_to_file(filename, A, B):
    try:
        L, U = lu_decomposition(A)
        X = solve_lu(L, U, B)
        
        # Lưu kết quả vào file
        with open(filename, 'w') as f:
            f.write("Ma trận A:\n")
            np.savetxt(f, A, fmt='%.2f')
            f.write("\nMa trận L (tam giác dưới):\n")
            np.savetxt(f, L, fmt='%.2f')
            f.write("\nMa trận U (tam giác trên):\n")
            np.savetxt(f, U, fmt='%.2f')
            f.write("\nVector B:\n")
            np.savetxt(f, B, fmt='%.2f')
            f.write("\nNghiệm của hệ phương trình AX = B:\n")
            np.savetxt(f, X, fmt='%.2f')
            
        print(f"Đã lưu kết quả vào file: {filename}")
        
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    # Tạo một ma trận vuông cấp 7 cụ thể và một vector B
    A = np.array([
        [10, -7, 0, 1, 2, 3, 5],
        [-3, 2, -3, 0, -5, 4, 9],
        [5, 1, 14, 3, 7, 2, -4],
        [0, -2, 3, 13, -1, 2, 1],
        [6, 0, -2, 3, 15, 2, 5],
        [4, 5, 3, -1, 3, 18, 1],
        [2, 1, 6, 4, 3, 2, 11]
    ], dtype=float)

    B = np.array([1, 2, 3, 4, 5, 7, 7], dtype=float)

    # Lưu kết quả vào file
    path = "/Users/a8888/Documents/doc uni/gts/Numerical-Analysis-Code-main/Matrix_Equations/LU/lu_matrix_solution.txt"
    save_solution_to_file(path, A, B)
