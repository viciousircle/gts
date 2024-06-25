import numpy as np

def jordan_form(A):
    """
    Tính dạng Jordan chuẩn của ma trận A.
    Trả về ma trận P và ma trận chuẩn tắc Jordan J.
    """
    eigenvalues, P = np.linalg.eig(A)
    J = np.diag(eigenvalues)  # Ma trận Jordan ban đầu
    return P, J

def eigen_decomposition(A):
    """
    Tính các giá trị riêng và các vector riêng của ma trận A
    bằng phương pháp Danielevski.
    """
    P, J = jordan_form(A)
    n = A.shape[0]
    eigenvalues = np.zeros(n)
    eigenvectors = np.zeros((n, n))
    
    for i in range(n):
        lambda_i = J[i, i]
        block_size = 1
        
        # Xác định kích thước khối Jordan
        if i < n - 1 and J[i, i+1] == 1:
            block_size += 1
        
        # Giá trị riêng là phần tử trên đường chéo của khối Jordan
        eigenvalues[i] = lambda_i
        
        # Vector riêng tương ứng với giá trị riêng lambda_i
        v = P[:, i]
        
        # Tính toán vector riêng cho khối Jordan
        for _ in range(1, block_size):
            v = np.dot(A - lambda_i * np.eye(n), v)
        
        eigenvectors[:, i] = v / np.linalg.norm(v)
    
    return eigenvalues, eigenvectors

# Ví dụ minh họa cho ma trận 6x6
if __name__ == "__main__":
    A = np.array([[4, 1, 2, 0, 0, 0],
                  [1, 5, 3, 0, 0, 0],
                  [2, 3, 6, 0, 0, 0],
                  [0, 0, 0, 2, 1, 0],
                  [0, 0, 0, 1, 3, 0],
                  [0, 0, 0, 0, 0, 5]])
    
    print("Ma trận A:")
    print(A)
    
    print("\nBước 1: Tính ma trận chuẩn tắc Jordan của ma trận A")
    P, J = jordan_form(A)
    print("   Ma trận P:")
    print(P)
    print("   Ma trận chuẩn tắc Jordan J:")
    print(J)
    
    print("\nBước 2: Tính các giá trị riêng và các vector riêng từ ma trận chuẩn tắc Jordan")
    eigenvalues, eigenvectors = eigen_decomposition(A)
    
    print("   Các giá trị riêng của A:")
    for i, eigenvalue in enumerate(eigenvalues):
        print(f"   λ_{i+1} =", eigenvalue)
    
    print("\n   Các vector riêng tương ứng:")
    for i, eigenvector in enumerate(eigenvectors.T):
        print(f"   v_{i+1} =", eigenvector)
