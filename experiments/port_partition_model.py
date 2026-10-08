"""Arm-token graph encoder with exact finite-partition decoding."""
import torch
from torch import nn
from experiments.port_partition import PAIRS, RELATIONS


def encoder():
    return nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.ReLU(),
                         nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten())


class PortPartitionNet(nn.Module):
    def __init__(self, graph=True, independent=False, message_passing=True):
        super().__init__()
        self.graph = graph
        self.independent = independent
        self.message_passing = message_passing
        self.center = encoder()
        if graph:
            self.arm = encoder()
            self.attention = nn.MultiheadAttention(32, 4, batch_first=True)
            self.norm = nn.LayerNorm(32)
            self.edge = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        else:
            self.head = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 6))
        self.register_buffer('relations', torch.tensor(RELATIONS))

    def forward(self, image):
        center = self.center(image)
        if not self.graph:
            return self.head(center)
        # Canonical outward strips: all four arm crops share one encoder.
        strips = [image[:, :, 18:30, :24].flip(-1), image[:, :, 18:30, 24:],
                  image[:, :, :24, 18:30].transpose(-1, -2).flip(-1),
                  image[:, :, 24:, 18:30].transpose(-1, -2)]
        nodes = torch.stack([self.arm(s) for s in strips], dim=1)
        if not self.independent and self.message_passing:
            for _ in range(2):
                messages, _ = self.attention(nodes, nodes, nodes, need_weights=False)
                nodes = self.norm(nodes + messages)
        edges = []
        for a, b in PAIRS:
            feature = torch.cat([nodes[:, a]+nodes[:, b],
                                 (nodes[:, a]-nodes[:, b]).abs(),
                                 nodes[:, a]*nodes[:, b], center], dim=1)
            edges.append(self.edge(feature))
        unary = torch.cat(edges, dim=1)
        if self.independent:
            return unary
        # Scores over all 15 legal equivalence relations. MAP is always transitive.
        return unary @ self.relations.T
