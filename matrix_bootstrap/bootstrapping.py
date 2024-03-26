import os, sys
import numpy as np
import matplotlib.pyplot as plt


def map(f):
    """Override mapping function to handle tuples"""

    def curry(t):
        return (f(t[0]), f(t[1]))

    return curry


def the(x):
    """Identity function."""
    return x


class Pipe:
    """Stateless flow wrapper.

    To get the result, in some situations, a sequential call of several functions
    is used, for example,
        result = a(b(c(d(e(data)))))
    The disadvantage of this syntax is that it should be read from right to left,
    like Hebrew. An alternative to this: using variables to store intermediate results:
        e_result = e(data)
        d_result = d(e_result)
        ...
        result = a(b_result)
    which is also not good because it creates unnecessary variables.
    The flow approach has no drawbacks. To build a flow, a wrapper class is used, in which
    two simple operators are defined: the pipe operator >> and the unwrap operator |.
    First, the data is wrapped in the wrapper class Pipe(data) and then sent to the pipe >>. So, now
    the computational expression is like a chain of transformations
        Pipe(data) >> e >> d >> c >> b >> a
    which is written more naturally in terms of data processing sequence.
    Each stage of the pipe wraps the result in the wrapper class, therefore in order to
    get an explicit result at the end of the pipe apply the unwrap operator to
    the last function called
        result = Pipe(data) >> e >> d >> c >> b | a
    or use the identity function
        result = Pipe(data) >> e >> d >> c >> b >> a | the
    """

    def __init__(self, val, run_if: bool = True):
        self._val = val
        self.run = run_if

    def __rshift__(self, other):
        """Pipe operator >>."""
        return Pipe(other(self._val), run_if=self.run) if self.run else self

    def __lshift__(self, other):
        """Bind operator << for debugging purposes."""
        result = other(self._val)
        print(result)
        return Pipe(result, run_if=self.run) if self.run else self

    def __or__(self, other):
        """Unwrap operator |."""
        return other(self._val) if self.run else self._val


def load_matrix(path):
    """Charge une matrice à partir d'un fichier texte."""
    return np.loadtxt(path, delimiter=" ")


def gram_schmidt(A):
    Q, _ = np.linalg.qr(A)  # QR decomposition of matrix A
    return Q


def build_bootstrap_sample(orthonormal_basis):
    def curry(qd_decompositions):
        # qd_decompositions = (eigenvectors, eigenvalues) tuples
        n = qd_decompositions[0][0].size
        n_bootstrap = 2**n
        basis_processor = gram_schmidt if orthonormal_basis else the

        def inner_int_to_binary_list(k):
            """The `int_to_binary_list` function takes a positive integer representing the number of eigenvectors in
            two matrices and generates a list of binary representations.
            These binary representations can be used to determine all possible combinations of eigenvectors from both matrices.
            0 means eigenvector from A, 1 means eigenvector from B"""
            # Convert integer to binary string, remove '0b' prefix
            binary_str = bin(k)[2:]
            binary_str_padded = binary_str.zfill(n)  # Pad with zeros to ensure length n
            mask_array = np.array(
                [int(bit) for bit in binary_str_padded]
            )  # Convert each character to integer
            return mask_array
        
        def inner_swap_eigenvectors(mask):
            eigenvecs = (
                Pipe([qd_decompositions[mask[i]][1][:, i] for i in range(len(mask))])
                >> np.column_stack
                | basis_processor  # if basis is set to be orthonormal do gram_schmidt, do nothing otherwise
            )
            eigenvals = [qd_decompositions[mask[i]][0][i] for i in range(len(mask))]
            return np.dot(
                np.dot(eigenvecs, np.diag(eigenvals)), np.transpose(eigenvecs)
            )

        return [
            Pipe(i) >> inner_int_to_binary_list | inner_swap_eigenvectors
            for i in range(n_bootstrap)
        ]

    return curry


def collect_statistics(samples):
    def inner_FoM(sample):
        C_sub = np.linalg.inv(sample)[2:4, 2:4]
        return np.sqrt(1.0 / (np.linalg.det(C_sub)))

    average_matrix = np.average(samples, axis=0)
    FoM_stat = [inner_FoM(sample) for sample in samples]
    return average_matrix, FoM_stat


def plot_FoM_histogram(data):
    plt.hist(data[1], bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("FoM Histogram")
    plt.show()
    return data


def save_results(F_final):
    # Inverser la matrice de Fisher finale pour obtenir la matrice de covariance
    F_final_inverse = np.linalg.inv(F_final[0])

    # Sauvegarder la matrice de Fisher finale et son inverse
    np.savetxt("F_final.txt", F_final[0])
    np.savetxt("F_final_inverse.txt", F_final_inverse)
    return F_final


# the main flow
F_final = (
    Pipe(("r:/F1.txt", "r:/F2.txt"))
    >> map(load_matrix)
    >> map(np.linalg.eig)
    >> build_bootstrap_sample(orthonormal_basis=True)
    >> collect_statistics
    >> plot_FoM_histogram
    >> save_results
    | the
)[0]

# Remplacer la commande suivante par la manière dont vous utilisez printFoM dans votre environnement
os.system("python matrix_bootstrap/printFoM.py F_final.txt OPT GCph 0 0 F N")
