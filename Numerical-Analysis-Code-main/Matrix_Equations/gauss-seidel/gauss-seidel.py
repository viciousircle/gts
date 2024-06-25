import numpy as np

def check_diagonal_dominance(A):
    # Function to check if matrix A is diagonally dominant
    n = A.shape[0]
    for i in range(n):
        row_sum = np.sum(np.abs(A[i,:])) - np.abs(A[i,i])
        if np.abs(A[i,i]) <= row_sum:
            return False
    return True

def gauss_seidel(A, B, eps, max_iter=1000):
    # Gauss-Seidel method to solve AX = B with given tolerance eps
    n = A.shape[0]
    X = np.zeros(n)  # Initial guess
    
    if not check_diagonal_dominance(A):
        print("Ma trận không thỏa mãn điều kiện chéo trội hàng.")
        return None, []
    
    iteration = 0
    output = []
    while iteration < max_iter:
        X_new = np.zeros(n)
        for i in range(n):
            s1 = np.dot(A[i, :i], X_new[:i])
            s2 = np.dot(A[i, i+1:], X[i+1:])
            X_new[i] = (B[i] - s1 - s2) / A[i, i]
        
        # Store current iteration result
        output.append(X_new.copy())
        
        # Check convergence
        if np.linalg.norm(X_new - X, ord=np.inf) < eps:
            print(f"Converged after {iteration + 1} iterations.")
            output.append(X_new.copy())
            return X_new, output
        
        X = X_new
        iteration += 1
    
    print(f"Did not converge after {max_iter} iterations.")
    return None, output

def save_output_to_file(output, filename):
    try:
        with open(filename, 'w') as f:
            f.write("Gauss-Seidel iterative solver output:\n")
            for i, x in enumerate(output):
                if i < 3 or i >= len(output) - 3:
                    f.write(f"Iteration {i + 1}: {x}\n")
            print(f"Output saved to {filename}.")
    except Exception as e:
        print(f"Error saving output to {filename}: {e}")
        raise

if __name__ == "__main__":
    # Đọc ma trận từ file input
    input_filename = "/Users/a8888/Documents/doc uni/gts/Numerical-Analysis-Code-main/Matrix_Equations/gauss-seidel/matrix_input.txt"
    try:
        matrix = np.loadtxt(input_filename)
        A = matrix[:, :-1]
        B = matrix[:, -1]
    except Exception as e:
        print(f"Error reading matrix from {input_filename}: {e}")
        exit(1)
    
    eps = 1e-6   # Sai số
    max_iter = 1000  # Số lần lặp tối đa
    
    # Giải hệ phương trình bằng phương pháp lặp Gauss-Seidel
    solution, output = gauss_seidel(A, B, eps, max_iter)
    
    if solution is not None:
        print("Nghiệm gần đúng:")
        print(solution)
        print("Kiểm tra kết quả:")
        print(np.dot(A, solution))
        
        # Lưu output vào file
        output_filename = "/Users/a8888/Documents/doc uni/gts/Numerical-Analysis-Code-main/Matrix_Equations/gauss-seidel/matrix_output.txt"
        save_output_to_file(output, output_filename)
