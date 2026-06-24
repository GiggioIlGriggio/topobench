r"""Message Passing Simplicial Network (MPSN) backbone.

Native implementation of the boundary + upper-adjacency, GIN-style MPSN variant.

Notes
-----
**Scientific motivation.** Plain message-passing GNNs are bounded by the
1-Weisfeiler-Lehman (1-WL) test and *provably cannot count triangles* (Bodnar et
al. 2021, Sec. 1). MPSN lifts each triangle to a 2-simplex (clique lifting) and
runs message passing over the simplicial complex. Its colour-refinement
procedure, Simplicial-WL (SWL), is strictly more powerful than 1-WL and no less
powerful than 3-WL (arXiv:2103.03212, Thms. 5-6). With *injective*
aggregation/update functions (here in the GIN style of Xu et al., 2019), MPSN
matches SWL's distinguishing power, so triangles become first-class cells the
network can read and count.

**Variant implemented (the canonical efficient one, matching CIN).** For a
simplex :math:`\sigma` of rank :math:`r` we use boundary and upper-adjacency
messages only (co-boundary and lower-adjacency are dropped; the paper proves
boundary + upper alone retain full SWL power):

- boundary message (paper :math:`M_{\mathcal{B}}`), from :math:`\sigma`'s
  :math:`(r-1)`-faces,
  :math:`m_{\mathcal{B}}(\sigma) = \sum_{\tau \in \mathcal{B}(\sigma)} h_\tau`;
- upper-adjacency message (paper :math:`M_\uparrow`), from rank-:math:`r`
  simplices :math:`\tau` sharing a common :math:`(r+1)`-coface with
  :math:`\sigma`; the message also reads the shared coface's feature
  :math:`h_{\delta(\sigma,\tau)}`, giving
  :math:`m_\uparrow(\sigma) = \sum_{\tau \in \mathcal{N}_\uparrow(\sigma)}
  (h_\tau + h_{\delta(\sigma,\tau)})`;
- GIN-style injective update (paper's injective :math:`U`, à la Xu et al. 2019
  with a learnable per-rank :math:`\varepsilon`),
  :math:`h_\sigma^{t+1} = \mathrm{MLP}_r^t((1 + \varepsilon_r^t) h_\sigma^t
  + m_{\mathcal{B}}(\sigma) + m_\uparrow(\sigma))`.

**Routing operators.** We consume only the unsigned clique-lifted incidence
matrices :math:`B_1 \in \{0,1\}^{N_0 \times N_1}` (nodes :math:`\times` edges)
and :math:`B_2 \in \{0,1\}^{N_1 \times N_2}` (edges :math:`\times` triangles).
The boundary, upper-adjacency, and shared-coface terms are all derived from these
(no Hodge Laplacians). The lifting is unsigned, so no orientation/sign handling
is needed: we use the incidence support directly (equivalently ``.abs()``;
already 0/1). Because the SWL/triangle-counting setting is unoriented, we
deliberately do not use orientation-equivariant odd activations; the activation
between MLP linears is ``ReLU``, the faithful choice for unoriented expressivity.

References
----------
Bodnar, Frasca, Wang, Otter, Montúfar, Liò, Bronstein,
"Weisfeiler and Lehman Go Topological: Message Passing Simplicial Networks",
ICML 2021 (Spotlight), arXiv:2103.03212.
"""

import torch


def _to_dense(matrix: torch.Tensor) -> torch.Tensor:
    """Return a dense, unsigned (0/1) version of an incidence matrix.

    The clique lifting emits sparse COO incidence matrices with ``signed=False``,
    i.e. already 0/1. We densify (toy/complex-sized routing operators) and take
    the absolute value so the support is used regardless of any sign convention.

    Parameters
    ----------
    matrix : torch.Tensor
        Sparse or dense incidence matrix.

    Returns
    -------
    torch.Tensor
        Dense unsigned incidence matrix.
    """
    if matrix.is_sparse:
        matrix = matrix.to_dense()
    return matrix.abs()


def _upper_adjacency(boundary_next: torch.Tensor) -> torch.Tensor:
    r"""Build the rank-:math:`r` upper-adjacency neighbour operator.

    Two rank-:math:`r` simplices are *upper-adjacent* when they share a common
    :math:`(r+1)`-coface. With :math:`B_{r+1}` the (unsigned) rank-:math:`r` to
    rank-:math:`(r+1)` incidence (shape ``[N_r, N_{r+1}]``), the co-adjacency
    counts are the **off-diagonal** of :math:`B_{r+1} B_{r+1}^\top`; the diagonal
    (a simplex's own coface count) is removed so a simplex is not its own
    neighbour.

    Parameters
    ----------
    boundary_next : torch.Tensor
        Dense unsigned incidence :math:`B_{r+1}` of shape ``[N_r, N_{r+1}]``.

    Returns
    -------
    torch.Tensor
        Dense ``[N_r, N_r]`` upper-adjacency operator (zero diagonal). Applying
        it to rank-:math:`r` features yields
        :math:`\sum_{\tau \in \mathcal{N}_\uparrow(\sigma)} h_\tau`.
    """
    adjacency = boundary_next @ boundary_next.transpose(0, 1)
    adjacency = adjacency - torch.diag(torch.diagonal(adjacency))
    return adjacency


class MPSNLayer(torch.nn.Module):
    r"""One MPSN layer: boundary + upper-adjacency messages, GIN-style update.

    Implements, for every rank :math:`r \in \{0,1,2\}` simultaneously, the update
    of arXiv:2103.03212 (boundary + upper variant):

    .. math::
       h_\sigma^{t+1} =
         \mathrm{MLP}_r^t\!\bigl(
           (1 + \varepsilon_r^t)\, h_\sigma^t
           + m_{\mathcal{B}}(\sigma) + m_\uparrow(\sigma)
         \bigr)

    Each rank owns a **distinct** MLP and a **learnable scalar**
    :math:`\varepsilon_r` (initialised to ``0.0``), matching the injective
    GIN-style aggregator of Xu et al. (2019). Ranks whose boundary or upper
    neighbourhood is empty simply omit that term (rank 0 has no boundary; rank 2
    has no upper-adjacency).

    Parameters
    ----------
    hidden_dim : int
        Feature width shared by all ranks.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # One distinct MLP per rank (0, 1, 2): Linear -> ReLU -> Linear, no norm.
        for r in range(3):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            setattr(self, f"mlp_{r}", mlp)
            # Learnable per-rank epsilon, init 0.0 (GIN-style injective update).
            setattr(
                self,
                f"eps_{r}",
                torch.nn.Parameter(torch.zeros(())),
            )

    def _boundary_message(
        self, boundary: torch.Tensor, x_face: torch.Tensor
    ) -> torch.Tensor:
        r"""Boundary message :math:`m_{\mathcal{B}}(\sigma)` (paper :math:`M_{\mathcal{B}}`).

        Sums the features of :math:`\sigma`'s :math:`(r-1)`-faces. With
        :math:`B_r` the rank-:math:`(r-1)` to rank-:math:`r` incidence
        (``[N_{r-1}, N_r]``), :math:`B_r^\top h_{r-1}` gathers, for each
        rank-:math:`r` simplex, the sum of its faces' features.

        Parameters
        ----------
        boundary : torch.Tensor
            Dense unsigned incidence :math:`B_r` of shape ``[N_{r-1}, N_r]``.
        x_face : torch.Tensor
            Features of the :math:`(r-1)`-faces, shape ``[N_{r-1}, hidden]``.

        Returns
        -------
        torch.Tensor
            Boundary message per rank-:math:`r` simplex, shape ``[N_r, hidden]``.
        """
        return boundary.transpose(0, 1) @ x_face

    def _upper_message(
        self,
        upper_adjacency: torch.Tensor,
        boundary_next: torch.Tensor,
        x_self: torch.Tensor,
        x_coface: torch.Tensor,
        coface_multiplicity: float,
    ) -> torch.Tensor:
        r"""Upper-adjacency message :math:`m_\uparrow(\sigma)` (paper :math:`M_\uparrow`).

        .. math::
           m_\uparrow(\sigma) =
             \sum_{\tau \in \mathcal{N}_\uparrow(\sigma)}
               \bigl( h_\tau + h_{\delta(\sigma,\tau)} \bigr)

        The neighbour term :math:`\sum_\tau h_\tau` is the upper-adjacency
        operator applied to the rank's own features. The **shared-coface** term
        :math:`\sum_\tau h_{\delta(\sigma,\tau)}` is computed exactly via the
        coface multiplicity: in a simple-graph clique complex with
        ``complex_dim = 2`` two upper-adjacent simplices share exactly one coface
        :math:`\delta`, so

        .. math::
           \sum_{\tau \in \mathcal{N}_\uparrow(\sigma)} h_{\delta(\sigma,\tau)}
             = \sum_{\delta \in \mathrm{cofaces}(\sigma)}
                 (|\mathrm{faces}_r(\delta)| - 1)\, h_\delta ,

        which gives multiplicity **1** for rank 0 (each edge-coface has 2 nodes)
        and multiplicity **2** for rank 1 (each triangle-coface has 3 edges).
        :math:`B_{r+1} h_{r+1}` gathers, per rank-:math:`r` simplex, the sum of
        its cofaces' features.

        Parameters
        ----------
        upper_adjacency : torch.Tensor
            Dense ``[N_r, N_r]`` upper-adjacency operator (zero diagonal).
        boundary_next : torch.Tensor
            Dense unsigned incidence :math:`B_{r+1}` of shape ``[N_r, N_{r+1}]``.
        x_self : torch.Tensor
            Features of the rank-:math:`r` simplices, shape ``[N_r, hidden]``.
        x_coface : torch.Tensor
            Features of the rank-:math:`(r+1)` cofaces, shape ``[N_{r+1}, hidden]``.
        coface_multiplicity : float
            :math:`(|\mathrm{faces}_r(\delta)| - 1)`: 1 for rank 0, 2 for rank 1.

        Returns
        -------
        torch.Tensor
            Upper message per rank-:math:`r` simplex, shape ``[N_r, hidden]``.
        """
        neighbour_term = upper_adjacency @ x_self
        coface_term = coface_multiplicity * (boundary_next @ x_coface)
        return neighbour_term + coface_term

    def _update(
        self, rank: int, x_self: torch.Tensor, message: torch.Tensor
    ) -> torch.Tensor:
        r"""GIN-style injective update for one rank.

        .. math::
           h_\sigma^{t+1} =
             \mathrm{MLP}_r^t\!\bigl( (1 + \varepsilon_r^t)\, h_\sigma^t + m \bigr)

        where :math:`m` is the sum of the available boundary and upper messages.

        Parameters
        ----------
        rank : int
            Rank :math:`r \in \{0,1,2\}`.
        x_self : torch.Tensor
            Current features of the rank-:math:`r` simplices.
        message : torch.Tensor
            Aggregated boundary + upper message for this rank.

        Returns
        -------
        torch.Tensor
            Updated rank-:math:`r` features.
        """
        eps = getattr(self, f"eps_{rank}")
        mlp = getattr(self, f"mlp_{rank}")
        return mlp((1.0 + eps) * x_self + message)

    def forward(self, x_all, incidence_all):
        r"""Apply one MPSN layer to all three ranks.

        Parameters
        ----------
        x_all : tuple of torch.Tensor
            Current features ``(x_0, x_1, x_2)``.
        incidence_all : tuple of torch.Tensor
            Dense unsigned incidences ``(B1, B2, upper_adj_0, upper_adj_1)`` where
            ``B1`` is nodes :math:`\times` edges, ``B2`` is edges :math:`\times`
            triangles, ``upper_adj_0`` / ``upper_adj_1`` are the precomputed
            rank-0 / rank-1 upper-adjacency operators.

        Returns
        -------
        tuple of torch.Tensor
            Updated features ``(x_0, x_1, x_2)``.
        """
        x_0, x_1, x_2 = x_all
        b1, b2, upper_adj_0, upper_adj_1 = incidence_all

        # ---- rank 0 (nodes): no boundary; upper-adjacency via shared edge ----
        msg_0 = self._upper_message(
            upper_adjacency=upper_adj_0,
            boundary_next=b1,
            x_self=x_0,
            x_coface=x_1,
            coface_multiplicity=1.0,  # each edge-coface has 2 nodes -> 2-1 = 1
        )
        new_x_0 = self._update(0, x_0, msg_0)

        # ---- rank 1 (edges): boundary = 2 nodes; upper via shared triangle ----
        msg_1 = self._boundary_message(b1, x_0)
        if x_2.shape[0] > 0:
            msg_1 = (
                msg_1
                + self._upper_message(
                    upper_adjacency=upper_adj_1,
                    boundary_next=b2,
                    x_self=x_1,
                    x_coface=x_2,
                    coface_multiplicity=2.0,  # each tri-coface has 3 edges -> 3-1 = 2
                )
            )
        new_x_1 = self._update(1, x_1, msg_1)

        # ---- rank 2 (triangles): boundary = 3 edges; upper is empty ----
        if x_2.shape[0] > 0:
            msg_2 = self._boundary_message(b2, x_1)
            new_x_2 = self._update(2, x_2, msg_2)
        else:
            new_x_2 = x_2

        return new_x_0, new_x_1, new_x_2


class MPSN(torch.nn.Module):
    r"""Message Passing Simplicial Network (boundary + upper-adjacency variant).

    Lifts node features to higher-rank cells with per-rank input projections,
    then stacks ``n_layers`` :class:`MPSNLayer` blocks. Each layer performs the
    injective, GIN-style simplicial update of arXiv:2103.03212 (boundary +
    upper-adjacency messages), whose SWL colour refinement is strictly more
    powerful than 1-WL and no weaker than 3-WL — so the 2-cells (triangles)
    become countable, first-class objects.

    Parameters
    ----------
    in_channels_all : tuple of int
        Input feature dimensions on ``(nodes, edges, triangles)``.
    hidden_dim : int
        Hidden width shared across all ranks (default 64).
    n_layers : int
        Number of stacked MPSN layers (default 3).
    """

    def __init__(
        self, in_channels_all, hidden_dim: int = 64, n_layers: int = 3
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Per-rank input projection to the shared hidden width.
        self.in_linear_0 = torch.nn.Linear(in_channels_all[0], hidden_dim)
        self.in_linear_1 = torch.nn.Linear(in_channels_all[1], hidden_dim)
        self.in_linear_2 = torch.nn.Linear(in_channels_all[2], hidden_dim)

        self.layers = torch.nn.ModuleList(
            MPSNLayer(hidden_dim) for _ in range(n_layers)
        )

    def forward(self, x_all, incidence_all):
        r"""Forward pass over the simplicial complex.

        Parameters
        ----------
        x_all : tuple of torch.Tensor
            Input features ``(x_0, x_1, x_2)`` on nodes, edges, triangles.
        incidence_all : tuple of torch.Tensor
            ``(incidence_1, incidence_2)``: the unsigned clique-lifted incidence
            matrices :math:`B_1` (nodes :math:`\times` edges) and :math:`B_2`
            (edges :math:`\times` triangles). May be sparse COO; densified and
            ``.abs()``-ed internally. Used **only** for routing.

        Returns
        -------
        tuple of torch.Tensor
            Final hidden states ``(x_0, x_1, x_2)``, each of width ``hidden_dim``.
        """
        x_0, x_1, x_2 = x_all
        b1 = _to_dense(incidence_all[0])
        b2 = _to_dense(incidence_all[1])

        # Precompute the rank-0 and rank-1 upper-adjacency operators once.
        upper_adj_0 = _upper_adjacency(b1)
        upper_adj_1 = _upper_adjacency(b2)

        x_0 = self.in_linear_0(x_0)
        x_1 = self.in_linear_1(x_1)
        x_2 = self.in_linear_2(x_2)

        packed_incidence = (b1, b2, upper_adj_0, upper_adj_1)
        x_all = (x_0, x_1, x_2)
        for layer in self.layers:
            x_all = layer(x_all, packed_incidence)

        return x_all
