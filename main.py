import numpy as np


class MatrixMagician:
    def __init__(self, matrix):
        self.matrix = matrix

    def traverse_row(self, row):
        return self.matrix[row]

    def traverse_column(self, column):
        return [row[column] for row in self.matrix]

    def traverse_diagonal(self):
        return [
            self.matrix[i][i] for i in range(min(len(self.matrix), len(self.matrix[0])))
        ]

    def traverse_reverse_diagonal(self):
        return [
            self.matrix[i][len(self.matrix) - 1 - i]
            for i in range(min(len(self.matrix), len(self.matrix[0])))
        ]

    def traverse_spiral(self):
        spiral = []
        if self.matrix:
            row_start = 0
            row_end = len(self.matrix) - 1
            column_start = 0
            column_end = len(self.matrix[0]) - 1
            while row_start <= row_end and column_start <= column_end:
                for i in range(column_start, column_end + 1):
                    spiral.append(self.matrix[row_start][i])
                row_start += 1
                for i in range(row_start, row_end + 1):
                    spiral.append(self.matrix[i][column_end])
                column_end -= 1
                if row_start <= row_end:
                    for i in range(column_end, column_start - 1, -1):
                        spiral.append(self.matrix[row_end][i])
                    row_end -= 1
                if column_start <= column_end:
                    for i in range(row_end, row_start - 1, -1):
                        spiral.append(self.matrix[i][column_start])
                    column_start += 1
        return spiral

    def add(self, other_matrix):
        return np.add(self.matrix, other_matrix)

    def subtract(self, other_matrix):
        return np.subtract(self.matrix, other_matrix)

    def multiply(self, other_matrix):
        return np.dot(self.matrix, other_matrix)

    def transpose(self):
        return np.transpose(self.matrix)

    def eigenvalues(self):
        return np.linalg.eigvals(self.matrix)

    def determinant(self):
        return np.linalg.det(self.matrix)

    def singular_value_decomposition(self):
        U, Sigma, V = np.linalg.svd(self.matrix)
        return {
            "U matrix": U,
            "Sigma matrix": np.diag(Sigma),
            "V matrix": V,
        }

    def pseudoinverse(self):
        return np.linalg.pinv(self.matrix)


def main():
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    traversal = MatrixMagician(matrix)
    print(f"Row: {traversal.traverse_row(1)}")
    print(f"Col: {traversal.traverse_column(1)}")
    print(f"Diag: {traversal.traverse_diagonal()}")
    print(f"Rev Diag: {traversal.traverse_reverse_diagonal()}")
    print(f"Spiral: {traversal.traverse_spiral()}")
    print(f"Add: \n{traversal.add(matrix)}")
    print(f"Subtract: \n{traversal.subtract(matrix)}")
    print(f"Multiply: \n{traversal.multiply(matrix)}")
    print(f"Transpose: \n{traversal.transpose()}")
    print(f"Eigen Vals: {traversal.eigenvalues()}")
    print(f"Determinant: {traversal.determinant()}")

    svd = traversal.singular_value_decomposition()
    for name, matrix in svd.items():
        print(f"{name}:\n{matrix}\n")

    print(f"Pseudoinverse: \n{traversal.pseudoinverse()}")


if __name__ == "__main__":
    main()
