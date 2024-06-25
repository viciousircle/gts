import numpy as np

def read_matrix_from_file(filename):
    try:
        matrix = np.loadtxt(filename)
        return matrix
    except Exception as e:
        print(f"Error reading matrix from {filename}: {e}")
        raise

def iterative_solver(A, B, X0, epsilon=1e-6, max_iter=1000):
    n = A.shape[0]
    X = X0.copy()
    X_prev = X0.copy()
    output = []

    for iter_count in range(max_iter):
        # Calculate V = AX - B
        V = np.dot(A, X) - B
        
        # Update X using the iterative formula
        X = np.linalg.solve(A, B + np.dot(V, X))
        
        # Store the current iteration result
        output.append(X.copy())
        
        # Check convergence criterion
        if np.linalg.norm(X - X_prev) < epsilon:
            output.append(X.copy())
            print(f"Converged after {iter_count + 1} iterations.")
            return X, output
        
        X_prev = X.copy()
    
    raise ValueError(f"Did not converge after {max_iter} iterations.")

def save_output_to_file(output, filename, A, B, X0, epsilon, max_iter):
    try:
        with open(filename, 'w') as f:
            f.write("Iterative solver output:\n\n")
            f.write(f"System of equations:\n")
            f.write(f"A:\n{A}\n")
            f.write(f"B:\n{B}\n")
            f.write(f"Initial guess (X0):\n{X0}\n\n")
            f.write(f"Convergence criteria:\n")
            f.write(f"Epsilon (tolerance): {epsilon}\n")
            f.write(f"Maximum iterations: {max_iter}\n\n")
            f.write(f"Detailed iteration results:\n")
            for i, x in enumerate(output):
                f.write(f"Iteration {i + 1}:\n")
                f.write(f"X_{i+1} = {x}\n")
                f.write(f"Residual (||AX - B||) = {np.linalg.norm(np.dot(A, x) - B)}\n\n")
            print(f"Output saved to {filename}.")
    except Exception as e:
        print(f"Error saving output to {filename}: {e}")
        raise

if __name__ == "__main__":
    # Đọc ma trận từ file input
    input_filename = "/Users/a8888/Documents/doc uni/gts/Numerical-Analysis-Code-main/Matrix_Equations/Lap don/matrix_input.txt"
    matrix = read_matrix_from_file(input_filename)
    
    if matrix is None:
        exit(1)
    
    n = matrix.shape[0]
    A = matrix[:, :-1]  # Lấy ma trận A từ ma trận input
    B = matrix[:, -1]   # Lấy vector B từ ma trận input
    X0 = np.zeros(n)    # Giá trị khởi tạo X0
    epsilon = 1e-6      # Độ chính xác epsilon
    max_iter = 1000     # Số lần lặp tối đa
    
    try:
        # Giải hệ phương trình bằng phương pháp lặp đơn
        X, output = iterative_solver(A, B, X0, epsilon, max_iter)
        
        # Lưu kết quả từng bước vào file output
        output_filename = "/Users/a8888/Documents/doc uni/gts/Numerical-Analysis-Code-main/Matrix_Equations/Lap don/matrix_output.txt"
        save_output_to_file(output, output_filename, A, B, X0, epsilon, max_iter)
        
        # In ra nghiệm cuối cùng
        print("\nNghiệm tìm được:")
        print(X)
        
    except ValueError as e:
        print(e)
