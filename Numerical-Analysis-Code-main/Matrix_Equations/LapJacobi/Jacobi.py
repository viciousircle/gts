import numpy as np

def Cheotroihang(A, b):
    for i in range(A.shape[0]):
        max_val = abs(A[i, i])
        for j in range(A.shape[0]):
            if i == j:
                continue
            max_val -= abs(A[i, j])
        if max_val <= 0:
            return False
        
    return True

def Cheotroicot(A, b):
    for i in range(A.shape[0]):
        max_val = abs(A[i, i])
        for j in range(A.shape[0]):
            if i == j:
                continue
            max_val -= abs(A[j, i])
        if max_val <= 0:
            return False
        
    return True

def Jacobi_method(A, b, eps):
    output = []  # List to store output for each iteration
    
    if Cheotroihang(A, b):
        output.append("Ma trận thỏa mãn chéo trội hàng\n")
        for i in range(A.shape[0]):
            b[i] = b[i]/A[i, i]
            A[i, :] = A[i,:]/A[i, i]
        
        alpha = np.eye(A.shape[0]) - A
        q = np.linalg.norm(alpha.T, 1)
        output.append(f"q = {q}\n")
        output.append(f"alpha = \n{alpha}\n")
        output.append(f"beta = {b}\n\n")
        
        x0 = np.zeros(len(b))
        x = x0
        k = 0
        while True:
            x = alpha @ x.T + b
            output.append(f"Lần lặp thứ {k+1}: \n{x}\n")
            saiso = q * np.linalg.norm(x - x0, ord=1) / (1 - q)
            output.append(f"Sai số: {saiso}\n\n")
            if saiso < eps:
                output.append(f"Converged after {k+1} iterations.")
                return x, output
            x0 = x 
            k += 1
    
    elif Cheotroicot(A, b):
        output.append("Ma trận thỏa mãn chéo trội cột\n")
        T = np.zeros(A.shape)
        max_a = abs(A[0, 0])
        min_a = abs(A[0, 0])
        for i in range(A.shape[0]):
            T[i, i] = 1 / A[i, i]
            if max_a < abs(A[i, i]):
                max_a = abs(A[i, i])
            if min_a > abs(A[i, i]):
                min_a = abs(A[i, i])
        Lambda = max_a / min_a
        alpha = np.eye(A.shape[0]) - T @ A
        beta = T @ b
        q = np.linalg.norm(alpha, 1)
        output.append(f"q = {q}\n")
        output.append(f"alpha = \n{alpha}\n")
        output.append(f"beta = {beta}\n\n")
        
        x0 = np.ones(len(b))
        x = x0
        k = 0
        while True:
            x = alpha @ x.T + beta
            output.append(f"Lần lặp thứ {k+1}: \n{x}\n")
            saiso = Lambda * q * np.linalg.norm(x - x0, ord=1) / (1 - q)
            output.append(f"Sai số: {saiso}\n\n")
            if saiso < eps:
                output.append(f"Converged after {k+1} iterations.")
                return x, output
            x0 = x 
            k += 1
    
    else:
        output.append("Ma trận không thỏa mãn chéo trội")
    
    return None, output


def write_output_to_file(output, filename):
    try:
        with open(filename, 'w') as f:
            f.writelines(output)
        print(f"Output saved to {filename}.")
    except Exception as e:
        print(f"Error saving output to {filename}: {e}")
        raise

if __name__ == "__main__":
    input_filename = "/Users/a8888/Documents/doc uni/gts/Numerical-Analysis-Code-main/Matrix_Equations/LapJacobi/matrix.txt"
    try:
        matrix = np.loadtxt(input_filename)
        A = matrix[:, :-1]
        b = matrix[:, -1]
        eps = 1e-10   # Sai số
        Sol, output = Jacobi_method(A, b, eps)
        output.append(f"Nghiệm gần đúng: {Sol}\n")
        output.append(f"Kiểm tra kết quả: Ax = {np.dot(A, Sol)}\n")
        
        output_filename = "/Users/a8888/Documents/doc uni/gts/Numerical-Analysis-Code-main/Matrix_Equations/LapJacobi/matrix_output.txt"
        write_output_to_file(output, output_filename)
        
        print(f"\nNghiệm gần đúng: {Sol}")
        print(f"Kiểm tra kết quả: Ax = {np.dot(A, Sol)}")
    
    except Exception as e:
        print(e)
