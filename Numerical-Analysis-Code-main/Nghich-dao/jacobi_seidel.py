import numpy as np

def read_matrix_from_file(filename):
    try:
        matrix = np.loadtxt(filename)
        return matrix
    except Exception as e:
        print(f"Lỗi khi đọc ma trận từ file {filename}: {e}")
        return None

def save_matrix_to_file(matrix, filename):
    try:
        with open(filename, 'w') as f:
            np.savetxt(f, matrix, fmt='%.6f', delimiter=' ')
            print(f"Ma trận đã được lưu vào file {filename}.")
    except Exception as e:
        print(f"Lỗi khi lưu ma trận vào file {filename}: {e}")

def jacobi_iteration(A, tol_abs=1e-10, max_iter=1000):
    n = A.shape[0]
    D = np.diag(np.diag(A))
    LpU = A - D
    X = np.eye(n)  # Ma trận khởi tạo là đơn vị
    
    for k in range(max_iter):
        X_new = np.linalg.inv(D) @ (np.eye(n) - LpU) @ X
        
        if np.linalg.norm(X_new - X, ord=np.inf) < tol_abs:
            break
        
        X = X_new
    
    return X

def gauss_seidel_iteration(A, tol_abs=1e-10, max_iter=1000):
    n = A.shape[0]
    DmL = np.tril(A)
    U = A - DmL
    X = np.eye(n)  # Ma trận khởi tạo là đơn vị
    
    for k in range(max_iter):
        X_new = np.linalg.inv(DmL) @ (np.eye(n) - U) @ X
        
        if np.linalg.norm(X_new - X, ord=np.inf) < tol_abs:
            break
        
        X = X_new
    
    return X

if __name__ == "__main__":
    
    input_filename = "/Users/a8888/Documents/doc uni/gts/Numerical-Analysis-Code-main/Nghich-dao/matrix_input.txt"
    output_filename_jacobi = "/Users/a8888/Documents/doc uni/gts/Numerical-Analysis-Code-main/Nghich-dao/matrix_output_jacobi.txt"
    output_filename_gauss_seidel = "/Users/a8888/Documents/doc uni/gts/Numerical-Analysis-Code-main/Nghich-dao/matrix_output_gauss_seidel.txt"

    A = read_matrix_from_file(input_filename)
    
    if A is None:
        exit(1)
    
    print("Ma trận đọc từ file:")
    print(A)
    
    try:
        A_inv_jacobi = jacobi_iteration(A)
        print("\nMa trận nghịch đảo của A bằng phương pháp Jacobi:")
        print(A_inv_jacobi)
        save_matrix_to_file(A_inv_jacobi, output_filename_jacobi)
    except Exception as e:
        print(f"Lỗi khi tính nghịch đảo bằng Jacobi: {e}")
    
    try:
        A_inv_gauss_seidel = gauss_seidel_iteration(A)
        print("\nMa trận nghịch đảo của A bằng phương pháp Gauss-Seidel:")
        print(A_inv_gauss_seidel)
        save_matrix_to_file(A_inv_gauss_seidel, output_filename_gauss_seidel)
    except Exception as e:
        print(f"Lỗi khi tính nghịch đảo bằng Gauss-Seidel: {e}")
