"""
Author: Dimas Ahmad
Description: This file contains the model classes for the project.
"""

import torch


class STHCW(torch.nn.Module):
    def __init__(self, n, m, p, q, k=3, embedding_dim=32, weighted=False):
        super(STHCW, self).__init__()
        self.n = n
        self.m = m
        self.p = p
        self.q = q
        self.k = k
        self.embedding_dim = embedding_dim
        self.weighted = weighted

        self.W0 = torch.nn.Parameter(torch.randn(n + m + p + q, embedding_dim) * 0.01)
        self.W_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(embedding_dim, embedding_dim) * 0.01) for _ in range(k)]
        ) if weighted else None

        self.alpha = torch.nn.Parameter(torch.ones(k+1) /(k+1))

    def forward(self, A):
        E0 = torch.sparse.mm(A, self.W0)
        E_layers = [E0]
        E = E0

        for i in range(self.k):
            E = torch.sparse.mm(A, E)
            if self.weighted and self.W_list is not None:
                E = E @ self.W_list[i]
            E_layers.append(E)

        alpha = torch.nn.functional.softmax(self.alpha, dim=0)
        E_final = torch.zeros_like(E0)
        for i in range(self.k + 1):
            E_final += alpha[i] * E_layers[i]

        return E_final
