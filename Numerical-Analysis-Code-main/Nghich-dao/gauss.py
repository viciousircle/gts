import numpy as np

def gaussian_elimination(A):
    n = A.shape[0]
    I = np.eye(n)  # Ma trận đơn vị kích thước n x n
    
    # Nối A với I để tạo ma trận mở rộng [A | I]
    augmented_matrix = np.hstack((A, I))
    
    for i in range(n):
        # Chọn hàng chính để đảm bảo A[i, i] khác 0
        if augmented_matrix[i, i] == 0:
            for j in range(i + 1, n):
                if augmented_matrix[j, i] != 0:
                    augmented_matrix[[i, j]] = augmented_matrix[[j, i]]
                    break
        
        # Chia hàng i cho A[i, i] để đưa A[i, i] về thành 1
        augmented_matrix[i] /= augmented_matrix[i, i]
        
        # Biến đổi các hàng khác sao cho A[i, i] cột i là 0
        for j in range(n):
            if i != j:
                augmented_matrix[j] -= augmented_matrix[j, i] * augmented_matrix[i]
    
    # Ma trận nghịch đảo là các cột từ n+1 đến 2n của ma trận mở rộng
    inverse_A = augmented_matrix[:, n:]
    
    return inverse_A

def read_matrix_from_file(filename):
    try:
        matrix = np.loadtxt(filename)
        return matrix
    except Exception as e:
        print(f"Error reading matrix from {filename}: {e}")
        return None

if __name__ == "__main__":
    # Đọc ma trận từ file input
    input_filename = "/Users/a8888/Documents/doc uni/gts/Numerical-Analysis-Code-main/Nghich-dao/matrix_input.txt"
    matrix = read_matrix_from_file(input_filename)
    
    if matrix is not None:
        n = matrix.shape[0]
        if matrix.shape[0] != matrix.shape[1]:
            print("Ma trận không vuông, không thể tính nghịch đảo.")
        else:
            # Tìm ma trận nghịch đảo của A bằng phương pháp Gauss-Jordan
            A_inv = gaussian_elimination(matrix)
            
            # In ra ma trận nghịch đảo và kiểm tra tính chất A*A_inv = I
            print("Ma trận nghịch đảo của A:")
            print(A_inv)
            
            identity_matrix = np.dot(matrix, A_inv)
            print("\nKiểm tra: A * A_inv =")
            print(identity_matrix)
    else:
        print(f"Không thể đọc ma trận từ file {input_filename}.")
