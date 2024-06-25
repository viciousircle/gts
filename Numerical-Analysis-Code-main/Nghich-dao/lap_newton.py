import numpy as np

def inverse_newton_iteration(A, tol_abs=1e-6, tol_rel=1e-6, max_iter=100):
    n = A.shape[0]
    X = np.eye(n)  # Ước đoán ban đầu là ma trận đơn vị
    I = np.eye(n)
    iter_count = 0
    
    while iter_count < max_iter:
        # Tính sai số Delta
        Delta = X - np.eye(n)
        
        # Tính ma trận đạo hàm Jacobian
        J = np.dot(A, X) - I
        
        # Kiểm tra điều kiện dừng
        abs_error = np.linalg.norm(Delta)
        rel_error = abs_error / np.linalg.norm(X)
        
        if abs_error < tol_abs or rel_error < tol_rel:
            print(f"Đã đạt điều kiện dừng sau {iter_count} lần lặp.")
            return X
        
        # Cập nhật ma trận nghịch đảo bằng công thức lặp Newton
        try:
            X -= np.linalg.solve(J, Delta)
        except np.linalg.LinAlgError as e:
            print(f"Lỗi: {e}")
            return X
        
        iter_count += 1
    
    print("Không hội tụ sau số lần lặp tối đa.")
    return X

# Hàm đọc ma trận từ file
def read_matrix_from_file(filename):
    try:
        matrix = np.loadtxt(filename)
        return matrix
    except Exception as e:
        print(f"Lỗi khi đọc ma trận từ file {filename}: {e}")
        return None

# Hàm lưu ma trận vào file
def save_matrix_to_file(matrix, filename):
    try:
        with open(filename, 'w') as f:
            np.savetxt(f, matrix, fmt='%.6f', delimiter=' ')
            print(f"Ma trận đã được lưu vào file {filename}.")
    except Exception as e:
        print(f"Lỗi khi lưu ma trận vào file {filename}: {e}")

if __name__ == "__main__":
    input_filename = "/Users/a8888/Documents/doc uni/gts/Numerical-Analysis-Code-main/Nghich-dao/matrix_input.txt"
    output_filename = "/Users/a8888/Documents/doc uni/gts/Numerical-Analysis-Code-main/Nghich-dao/matrix_output.txt"

    A = read_matrix_from_file(input_filename)
    
    if A is None:
        exit(1)
    
    print("Ma trận đọc từ file:")
    print(A)
    
    try:
        A_inv = inverse_newton_iteration(A)
    except Exception as e:
        print(f"Lỗi trong quá trình tính toán: {e}")
        exit(1)
    
    print("\nMa trận nghịch đảo của A bằng phương pháp Newton:")
    print(A_inv)
    
    identity_matrix = np.dot(A, A_inv)
    print("\nKiểm tra: A * A_inv =")
    print(identity_matrix)
    
    save_matrix_to_file(A_inv, output_filename)
