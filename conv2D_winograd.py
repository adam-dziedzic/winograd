import sys
import torch
from torch import tensor


class Winograd(object):
    B = tensor(
        [[1.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, -1.0, 1.0],
         [-1.0, 1.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, -1.0]])
    B_T = B.transpose(1, 0)
    G = tensor(
        [[1.0, 0.0, 0.0],
         [0.5, 0.5, 0.5],
         [0.5, -0.5, 0.5],
         [0.0, 0.0, 1.0]])
    G_T = G.transpose(1, 0)
    A = tensor([[1.0, 0.0],
                [1.0, 1.0],
                [1.0, -1.0],
                [0.0, -1.0]])
    A_T = A.transpose(1, 0)

    def __init__(self, filter_value=None):
        super(Winograd, self).__init__()

        if filter_value is not None:
            self.filter = filter_value

    @staticmethod
    def main(input, filter):
        """
        Compute Winograd convolution.

        :param input:
        :param filter:
        :return: output
        """
        N, C, H, W = input.size()
        K, Cprime, r, rprime = filter.size()
        assert H == W
        assert r == rprime
        assert C == Cprime
        m = 2
        a = m + r - 1
        # TODO pad with zeros the input for perfect tiling and slice the output.
        if a % m != 0:
            raise Exception("Only the tiling of 4x4 of the input is accepted.")
        overlap = r - 1
        input = torch.transpose(input, 0, 1)
        assert input.size() == (C, N, H, W)
        # ntile = int(math.ceil(H//a))
        # P = N * ntile * ntile
        T = (W - a) // overlap + 1  # tiles_per_channel
        P = N * T * T
        U = torch.zeros(K, C, a, a)
        V = torch.zeros(C, P, a, a)
        for k in range(K):
            for c in range(C):
                U[k, c] = torch.matmul(Winograd.G,
                                       torch.matmul(filter[k, c], Winograd.G_T))
        for n in range(N):
            for tH in range(T):
                for tW in range(T):
                    for c in range(C):
                        b = n * (T * T) + tH * T + tW
                        vH = tH * (r - 1)
                        vW = tW * (r - 1)
                        V[c, b] = torch.matmul(Winograd.B_T, torch.matmul(
                            input[c, n, vH:vH + a, vW:vW + a], Winograd.B))
        M = torch.zeros(K, P, a, a)
        for k in range(K):
            for b in range(P):
                for c in range(C):
                    M[k, b] += U[k, c] * V[c, b]
        out_size = H - r + 1
        Y = torch.zeros(K, N, out_size, out_size)
        for k in range(K):
            for n in range(N):
                for tH in range(T):
                    for tW in range(T):
                        b = n * (T * T) + tH * T + tW
                        oH = tH * m
                        oW = tW * m
                        Y[k, n, oH:oH + m, oW:oW + m] = torch.matmul(
                            Winograd.A_T, torch.matmul(M[k, b], Winograd.A))

        Y = torch.transpose(Y, 0, 1)
        return Y

    @staticmethod
    def winograd_F_2_3(input, filter):
        """
        Compute winograd convolution with output of size 2x2 and filter of size
        3x3.

        :param input: 4x4
        :param filter: 3x3
        :return: 2x2
        """
        U = torch.matmul(Winograd.G, torch.matmul(filter, Winograd.G_T))
        V = torch.matmul(Winograd.B_T, torch.matmul(input, Winograd.B))
        return torch.matmul(Winograd.A_T, torch.matmul(U * V, Winograd.A))

    @staticmethod
    def winograd_F_1_3(input, filter):
        """
        Compute winograd convolution with output of size 1x1 and filter of size
        3x3. Input size is 3x3.

        :param input: 3x3
        :param filter: 3x3
        :return: 2x2
        """
        return input * filter


if __name__ == "__main__":
    import doctest

    sys.exit(doctest.testmod()[0])
