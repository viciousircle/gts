import numpy as np

def inverse_bordering_method(A):
    n = A.shape[0]
    
    # Bắt đầu với ma trận nghịch đảo của ma trận 1x1
    inv_A = np.array([[1 / A[0, 0]]])
    
    for k in range(1, n):
        # Xác định ma trận và vectơ tương ứng với viền thêm vào
        B = A[:k, :k]
        b = A[:k, k]
        c = A[k, :k]
        d = A[k, k]
        
        # Tính phần bổ sung của ma trận nghịch đảo
        u = -inv_A @ b
        v = -c @ inv_A
        alpha = d + c @ inv_A @ b
        
        if alpha == 0:
            raise ValueError("Matrix is singular and cannot be inverted using bordering method.")
        
        # Cập nhật ma trận nghịch đảo
        new_inv_A = np.zeros((k+1, k+1))
        new_inv_A[:k, :k] = inv_A + (u[:, None] @ v[None, :]) / alpha
        new_inv_A[:k, k] = u / alpha
        new_inv_A[k, :k] = v / alpha
        new_inv_A[k, k] = 1 / alpha
        
        inv_A = new_inv_A
    
    return inv_A

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
    
    try:
        A_inv = inverse_bordering_method(A)
    except ValueError as e:
        print(f"Lỗi: {e}")
        exit(1)
    
    print("\nMa trận nghịch đảo của A:")
    print(A_inv)
    
    identity_matrix = np.dot(A, A_inv)
    print("\nKiểm tra: A * A_inv =")
    print(identity_matrix)
    
    output_filename = "/Users/a8888/Documents/doc uni/gts/Numerical-Analysis-Code-main/Nghich-dao/matrix_output.txt"
    save_matrix_to_file(A_inv, output_filename)
