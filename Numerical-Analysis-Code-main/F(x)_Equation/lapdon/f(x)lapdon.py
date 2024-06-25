import numpy as np
import matplotlib.pyplot as plt
import math
import os

num_decimal = 7  # Số chữ số thập phân xuất hiện trong output
delta = 1e-7
MAXLOOP = 1e+7  # Số lần lặp tối đa
eta = 1e-3

def f(x):  # Hàm f(x)
    return 0.01 * x**3 - 2 * x + 1
    # return np.log(x) - 1

def g(x):  # Hàm g(x)/phi(x)
    return (0.01 * x**3 + 1) / 2
    # return x * np.log(x)

def dg(x):
    return (g(x + delta) - g(x - delta)) / (2 * delta)

def d2g(x):
    return (dg(x + delta) - dg(x - delta)) / (2 * delta)

def show_fx():
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Đồ thị biểu diễn phương trình x = g(x) ")
    x = np.linspace(b, a, 1000)
    plt.plot(x, g(x))
    plt.plot(x, x)
    plt.plot(0, 0, '+')
    plt.grid()
    plt.show()

def menu():
    print("")
    print(" " * 21, "_" * 45)
    print(" " * 20, "|{0:45}|".format(" "))
    print(" " * 20, "|{0:^9}{1:^9}{0:^15}|".format(" ", "1. Sai số tiên nghiệm"))
    print(" " * 20, "|{0:^9}{1:^9}{0:^16}|".format(" ", "2. Sai số hậu nghiệm"))
    print(" " * 20, "|{0:45}|".format(" "))
    print(" " * 21, "-" * 45)
    print("")
    print("")

def gradient_descent(a, b, sign):
    x_ct = a
    x_new = x_ct - sign * eta * d2g(x_ct)
    count = 1
    while not (abs(d2g(x_ct)) < delta):
        x_ct = x_new
        x_new = x_ct - sign * eta * d2g(x_ct)
        count += 1
        if count > 1e+7:
            break
        if x_ct > b:
            return x_ct
        elif x_ct < a:
            return x_ct
    return x_ct

def find_Extreme_of_theFuntion(left, right):
    count = 2
    extreme = [left, right]
    while True:
        temp1 = gradient_descent(left, right, 1)
        temp2 = gradient_descent(left, right, -1)
        if (temp1 > right and temp2 < left) or (temp1 < left and temp2 > right):
            break
        if (temp1 > left) and (temp1 < right):
            extreme.append(temp1)
            left = extreme[count] + 0.05
            count += 1

        if (temp2 > left):
            if (temp2 < right):
                extreme.append(temp2)
                left = extreme[count] + 0.05
                count += 1
    return extreme

def check_input(left, right):
    global q
    if left > right:
        left, right = right, left
    ext = find_Extreme_of_theFuntion(left, right)
    dfext = [abs(dg(i)) for i in ext]  # Lưu giá trị đặc biệt của f'(x) để tìm MAX(f')
    q = max(dfext)
    if q < 1:
        return True
    else:
        print("q = ", q, " >= 1")
    return False

def lapdon(x0, q, num):
    x_old = x0  # xấp xỉ nghiệm đầu = cực trái
    x = g(x_old)  # kiểm tra lần đầu để xác định sai số

    if num == 1:
        n = int(np.ceil(math.log((1 - q) * eps / abs(x - x_old), q)))
        fout.write("\n" + "_" * 65)
        fout.write("\n|{0:^10}|{1:^25}|{2:^25}|".format("Lần lặp", "Nghiệm x", "f(x)", "Sai số"))
        fout.write("\n" + "_" * 65)
        fout.write("\n|{0:^10}|{1:^25}|{2:^25}|".format(1, round(x, num_decimal), round(f(x), num_decimal)))
        for count in range(2, n + 1):
            x = g(x)
            fout.write("\n|{0:^10}|{1:^25}|{2:^25}|".format(count, round(x, num_decimal), round(f(x), num_decimal)))
            if f(x) == 0:
                print("\n Nghiệm đúng của phương trình f(x) = 0 là : ", x)
                break
        fout.write("\n" + "_" * 65)

    elif num == 2:
        Delta = q * abs(x - x_old) / (1 - q)
        count = 1
        fout.write("\n" + "_" * 85)
        fout.write("\n|{0:^10}|{1:^25}|{2:^25}|{3:^20}|".format("Lần lặp", "Nghiệm x", "f(x)", "Sai số"))
        fout.write("\n" + "_" * 85)
        fout.write("\n|{0:^10}|{1:^25}|{2:^25}|{3:^20}|".format(count, round(x, num_decimal), round(f(x), num_decimal),
                                                                round(Delta, num_decimal)))
        while not Delta < eps and count < MAXLOOP:
            x_old = x
            x = g(x)
            Delta = q * abs(x - x_old) / (1 - q)
            count += 1
            fout.write(
                "\n|{0:^10}|{1:^25}|{2:^25}|{3:^20}|".format(count, round(x, num_decimal), round(f(x), num_decimal),
                                                             round(Delta, num_decimal)))
        fout.write("\n" + "_" * 85)
    return [x, count]

if __name__ == "__main__":
    q = 1.0
    print("Nhập khoảng tìm nghiệm: ")
    a = float(input("Nhập a: "))
    b = float(input("Nhập b: "))
    eps = float(input("Nhập sai số: "))
    show_fx()
    check = check_input(a, b)
    if not check:
        print("Kiểm tra lại INPUT")
    elif check:
        menu()
        choose = int(input("Nhập lựa chọn: "))
        x0 = float(input("Xấp xỉ đầu: "))
        current_dir = os.path.dirname(__file__)
        if choose == 1:
            output_path = os.path.join(current_dir, "output_tn_ld.txt")
            fout = open(output_path, mode='w', encoding='utf-8')
            try:
                x = lapdon(x0, q, 1)
                if a < x[0] < b:
                    print("Nghiệm của phương trình: ", x[0])
                    print("Số lần lặp: ", x[1])
                    print("q = ", q)
                    fout.write(f"\nNghiệm của phương trình: {x[0]}")
                else:
                    print("Trong khoảng (", a, ",", b, ") không có nghiệm")
            finally:
                fout.close()
        elif choose == 2:
            output_path = os.path.join(current_dir, "output_hn_ld.txt")
            fout = open(output_path, mode='w', encoding='utf-8')
            try:
                x = lapdon(x0, q, 2)
                if a < x[0] < b:
                    print("Nghiệm của phương trình: ", x[0])
                    print("Số lần lặp: ", x[1])
                    print("q = ", q)
                    fout.write(f"\nNghiệm của phương trình: {x[0]}")
                else:
                    print("Trong khoảng (", a, ",", b, ") không có nghiệm")
            finally:
                fout.close()
