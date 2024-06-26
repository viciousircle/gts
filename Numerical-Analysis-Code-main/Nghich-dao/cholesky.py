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
                L[i, j] = (A[i, j] - sum_term) / L[j, j]
    
    return L

def inverse_from_cholesky(L):
    n = L.shape[0]
    I = np.eye(n)
    inverse_A = np.zeros((n, n))
    
    for i in range(n):
        Y = np.linalg.solve(L, I[:, i])
        inverse_A[:, i] = np.linalg.solve(L.T, Y)
    
    return inverse_A

def save_matrix_to_file(matrix, filename):
    try:
        with open(filename, 'w') as f:
            np.savetxt(f, matrix, fmt='%.6f', delimiter=' ')
            print(f"Ma trận đã được lưu vào file {filename}.")
    except Exception as e:
        print(f"Lỗi khi lưu ma trận vào file {filename}: {e}")

def read_matrix_from_file(filename):
    try:
        matrix = np.loadtxt(filename)
        return matrix
    except Exception as e:
        print(f"Lỗi khi đọc ma trận từ file {filename}: {e}")
        return None

if __name__ == "__main__":
    input_filename = "/Users/a8888/Documents/doc uni/gts/Numerical-Analysis-Code-main/Nghich-dao/matrix_input.txt"
    A = read_matrix_from_file(input_filename)
    
    if A is None:
        exit(1)
    
    print("Ma trận đọc từ file:")
    print(A)
    
    if not np.allclose(A, A.T):
        print("Lỗi: Ma trận không đối xứng.")
        exit(1)
    if not np.all(np.linalg.eigvals(A) > 0):
        print("Lỗi: Ma trận không xác định dương.")
        exit(1)
    
    L = cholesky_decomposition(A)
    
    A_inv = inverse_from_cholesky(L)
    
    print("\nMa trận nghịch đảo của A:")
    print(A_inv)
    
    identity_matrix = np.dot(A, A_inv)
    print("\nKiểm tra: A * A_inv =")
    print(identity_matrix)
    
    output_filename = "/Users/a8888/Documents/doc uni/gts/Numerical-Analysis-Code-main/Nghich-dao/matrix_output.txt"
    save_matrix_to_file(A_inv, output_filename)
