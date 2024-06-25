import numpy as np

def cholesky_decomposition(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1):
            sum_term = sum(L[i, k] * L[j, k] for k in range(j))
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - sum_term)
            else:
                L[i, j] = (1.0 / L[j, j] * (A[i, j] - sum_term))
    
    return L

def inverse_from_cholesky(L):
    n = L.shape[0]
    I = np.eye(n)
    inverse_A = np.zeros((n, n))
    
    # Giải hệ phương trình LL^T * X = I để tìm X = L^-T * L^-1
    for i in range(n):
        Y = np.zeros(n)
        for j in range(n):
            Y[j] = (I[i, j] - sum(L[i, k] * Y[k] for k in range(j))) / L[i, j]
        for j in range(n-1, -1, -1):
            inverse_A[j, i] = (Y[j] - sum(L[j, k] * inverse_A[k, i] for k in range(j+1, n))) / L[j, j]
    
    return inverse_A

def save_matrix_to_file(matrix, filename):
    try:
        with open(filename, 'w') as f:
            np.savetxt(f, matrix, fmt='%.6f', delimiter=' ')
            print(f"Ma trận đã được lưu vào file {filename}.")
    except Exception as e:
        print(f"Lỗi khi lưu ma trận vào file {filename}: {e}")

if __name__ == "__main__":
    # Sử dụng một ma trận cụ thể xác định dương và đối xứng
    A = np.array([[4, 1, 2],
                  [1, 5, 3],
                  [2, 3, 6]])
    
    # Kiểm tra xác định dương và đối xứng
    if not np.allclose(A, A.T):
        print("Lỗi: Ma trận không đối xứng.")
        exit(1)
    if not np.all(np.linalg.eigvals(A) > 0):
        print("Lỗi: Ma trận không xác định dương.")
        exit(1)
    
    # Phân tích Cholesky để tìm ma trận L
    L = cholesky_decomposition(A)
    
    # Tính ma trận nghịch đảo của A từ ma trận L
    A_inv = inverse_from_cholesky(L)
    
    # In ra ma trận nghịch đảo và kiểm tra tính chất A * A_inv = I
    print("\nMa trận nghịch đảo của A:")
    print(A_inv)
    
    identity_matrix = np.dot(A, A_inv)
    print("\nKiểm tra: A * A_inv =")
    print(identity_matrix)
    
    # Lưu ma trận nghịch đảo vào file
    output_filename = "matrix_output.txt"
    save_matrix_to_file(A_inv, output_filename)
