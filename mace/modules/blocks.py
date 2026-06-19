###########################################################################################
# Elementary Block for Building O(3) Equivariant Higher Order Message Passing Neural Network
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from abc import abstractmethod
from typing import Any, Callable, List, Dict, Optional, Tuple, Union
import torch
import numpy as np
import torch.nn.functional
from e3nn import nn, o3
from e3nn.util.jit import compile_mode

from mace.modules.wrapper_ops import (
    CuEquivarianceConfig,
    FullyConnectedTensorProduct,
    Linear,
    OEQConfig,
    SymmetricContractionWrapper,
    TensorProduct,
    TransposeIrrepsLayoutWrapper,
)
from mace.tools.compile import simplify_if_compile
from mace.tools.scatter import scatter_sum
from mace.tools.utils import LAMMPS_MP

from .irreps_tools import mask_head, reshape_irreps, tp_out_irreps_with_instructions
from .radial import (
    AgnesiTransform,
    BesselBasis,
    ChebychevBasis,
    GaussianBasis,
    PolynomialCutoff,
    RadialMLP,
    SoftTransform,
)


@compile_mode("script")
class LinearNodeEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        cueq_config: Optional[CuEquivarianceConfig] = None,
    ):
        super().__init__()
        self.linear = Linear(
            irreps_in=irreps_in, irreps_out=irreps_out, cueq_config=cueq_config
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:  # [n_nodes, irreps]
        return self.linear(node_attrs)


@compile_mode("script")
class LinearReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.linear = Linear(
            irreps_in=irreps_in, irreps_out=irrep_out, cueq_config=cueq_config
        )

    # before
    # def forward(
    #    self,
    #    x: torch.Tensor,
    #    heads: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
    #) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
    #    return self.linear(x)  # [n_nodes, 1]
    # for output conditioning also pass method and node batch
    def forward(
        self,
        x: torch.Tensor,
        heads: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
        method_z: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
        node_batch: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        return self.linear(x)


@compile_mode("script")
class NonLinearReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        num_heads: int = 1,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.num_heads = num_heads
        self.linear_1 = Linear(
            irreps_in=irreps_in, irreps_out=self.hidden_irreps, cueq_config=cueq_config
        )
        self.non_linearity = simplify_if_compile(nn.Activation)(
            irreps_in=self.hidden_irreps, acts=[gate]
        )
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps, irreps_out=irrep_out, cueq_config=cueq_config
        )

    # before
    #def forward(
    #    self, x: torch.Tensor, heads: Optional[torch.Tensor] = None
    #) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
    #    x = self.non_linearity(self.linear_1(x))
    #    if hasattr(self, "num_heads"):
    #        if self.num_heads > 1 and heads is not None:
    #            x = mask_head(x, heads, self.num_heads)
    #    return self.linear_2(x)  # [n_nodes, len(heads)]
    # output conditioining
    def forward(
        self,
        x: torch.Tensor,
        heads: Optional[torch.Tensor] = None,
        method_z: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
        node_batch: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        x = self.non_linearity(self.linear_1(x))
        if hasattr(self, "num_heads"):
            if self.num_heads > 1 and heads is not None:
                x = mask_head(x, heads, self.num_heads)
        return self.linear_2(x)

# continuous basis-mix readout block
@compile_mode("script")
class ContinuousBasisReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        method_dim: int,
        num_basis_heads: int = 4,
        mixer_hidden_dim: int = 0,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.method_dim = method_dim
        self.num_basis_heads = num_basis_heads
        self.mixer_hidden_dim = mixer_hidden_dim

        self.linear_1 = Linear(
            irreps_in=irreps_in,
            irreps_out=self.hidden_irreps,
            cueq_config=cueq_config,
        )
        self.non_linearity = simplify_if_compile(nn.Activation)(
            irreps_in=self.hidden_irreps,
            acts=[gate],
        )
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=o3.Irreps(f"{num_basis_heads}x0e"),
            cueq_config=cueq_config,
        )

        if mixer_hidden_dim > 0:
            self.mixer = torch.nn.Sequential(
                torch.nn.Linear(method_dim, mixer_hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(mixer_hidden_dim, num_basis_heads),
            )
            # initialize final layer to uniform logits at start
            torch.nn.init.zeros_(self.mixer[-1].weight)
            torch.nn.init.zeros_(self.mixer[-1].bias)
        else:
            self.mixer = torch.nn.Linear(method_dim, num_basis_heads)
            torch.nn.init.zeros_(self.mixer.weight)
            torch.nn.init.zeros_(self.mixer.bias)

    def forward(
        self,
        x: torch.Tensor,
        heads: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
        method_z: Optional[torch.Tensor] = None,
        node_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if method_z is None:
            raise ValueError(
                "ContinuousBasisReadoutBlock requires method_z in forward()."
            )
        if node_batch is None:
            raise ValueError(
                "ContinuousBasisReadoutBlock requires node_batch in forward()."
            )

        x = self.non_linearity(self.linear_1(x))
        basis_es = self.linear_2(x)  # [n_nodes, K]

        mix_logits = self.mixer(method_z.to(basis_es.dtype))  # [n_graphs, K]
        mix_weights = torch.softmax(mix_logits, dim=-1)
        mix_weights_nodes = mix_weights[node_batch]  # [n_nodes, K]

        node_e = torch.sum(basis_es * mix_weights_nodes, dim=-1, keepdim=True)
        return node_e  # [n_nodes, 1]


# FiLM Readout block
@compile_mode("script")
class ReadoutFiLMBlock(torch.nn.Module):
    """Final scalar readout with graph-level method FiLM conditioning.

    The block assumes that the final node features are scalar-only, which is
    already the case for the final MACE readout in your current model setup.

    Equations:
        h_i = act(W1 x_i)
        h_i' = (1 + gamma(z_m)) * h_i + beta(z_m)
        eps_i = W2 h_i'
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        method_dim: int,
        cueq_config: Optional[Dict[str, Any]] = None,
        oeq_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        del cueq_config, oeq_config  # not used in this simple scalar-only block

        irreps_in = o3.Irreps(irreps_in)
        MLP_irreps = o3.Irreps(MLP_irreps)
        self.hidden_irreps = MLP_irreps

        if method_dim is None or method_dim <= 0:
            raise ValueError("ReadoutFiLMBlock requires method_dim > 0.")

        if irreps_in.lmax > 0:
            raise ValueError(
                "ReadoutFiLMBlock expects scalar-only input irreps. "
                f"Got irreps_in={irreps_in}."
            )

        if MLP_irreps.lmax > 0:
            raise ValueError(
                "ReadoutFiLMBlock expects scalar-only MLP_irreps. "
                f"Got MLP_irreps={MLP_irreps}."
            )

        in_dim = irreps_in.count(o3.Irrep(0, 1))
        hidden_dim = MLP_irreps.count(o3.Irrep(0, 1))

        if in_dim <= 0:
            raise ValueError(f"Could not infer scalar input dim from {irreps_in}.")
        if hidden_dim <= 0:
            raise ValueError(f"Could not infer scalar hidden dim from {MLP_irreps}.")

        self.linear_1 = torch.nn.Linear(in_dim, hidden_dim)
        self.linear_2 = torch.nn.Linear(hidden_dim, 1)

        self.method_gamma = torch.nn.Linear(method_dim, hidden_dim)
        self.method_beta = torch.nn.Linear(method_dim, hidden_dim)

        # Identity FiLM initialization:
        # gamma(z)=0 and beta(z)=0, so initially h' = h.
        torch.nn.init.zeros_(self.method_gamma.weight)
        torch.nn.init.zeros_(self.method_gamma.bias)
        torch.nn.init.zeros_(self.method_beta.weight)
        torch.nn.init.zeros_(self.method_beta.bias)

        self.gate = gate

    def forward(
        self,
        x: torch.Tensor,
        method_z: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        if method_z is None:
            raise ValueError("ReadoutFiLMBlock requires method_z.")
        if node_batch is None:
            raise ValueError("ReadoutFiLMBlock requires node_batch.")

        h = self.linear_1(x)

        if self.gate is not None:
            h = self.gate(h)

        z_nodes = method_z[node_batch].to(dtype=h.dtype, device=h.device)

        gamma = 1.0 + self.method_gamma(z_nodes)
        beta = self.method_beta(z_nodes)

        h = gamma * h + beta
        return self.linear_2(h)


# delta readout film block
@compile_mode("script")
class DeltaReadoutFiLMBlock(torch.nn.Module):
    """Final scalar readout with a shared base readout plus method-conditioned FiLM correction.

    Equations:
        eps_i = eps_i_base(R) + delta_eps_i(R, z_m)

        eps_i_base = W2_base act(W1_base x_i)

        h_delta = act(W1_delta x_i)
        h_delta' = (1 + gamma(z_m)) * h_delta + beta(z_m)
        delta_eps_i = W2_delta h_delta'

    The correction branch is zero-initialized at the final layer, so initially:
        delta_eps_i = 0
    and the block behaves like a standard readout.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        method_dim: int,
        cueq_config: Optional[Dict[str, Any]] = None,
        oeq_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        del cueq_config, oeq_config  # unused in this simple scalar-only implementation

        irreps_in = o3.Irreps(irreps_in)
        MLP_irreps = o3.Irreps(MLP_irreps)
        
        self.hidden_irreps = MLP_irreps

        if method_dim is None or method_dim <= 0:
            raise ValueError("DeltaReadoutFiLMBlock requires method_dim > 0.")

        if irreps_in.lmax > 0:
            raise ValueError(
                "DeltaReadoutFiLMBlock expects scalar-only input irreps. "
                f"Got irreps_in={irreps_in}."
            )

        if MLP_irreps.lmax > 0:
            raise ValueError(
                "DeltaReadoutFiLMBlock expects scalar-only MLP_irreps. "
                f"Got MLP_irreps={MLP_irreps}."
            )

        in_dim = irreps_in.count(o3.Irrep(0, 1))
        hidden_dim = MLP_irreps.count(o3.Irrep(0, 1))

        if in_dim <= 0:
            raise ValueError(f"Could not infer scalar input dim from {irreps_in}.")
        if hidden_dim <= 0:
            raise ValueError(f"Could not infer scalar hidden dim from {MLP_irreps}.")

        self.gate = gate

        # Shared/base readout branch: standard geometry-only readout
        self.base_linear_1 = torch.nn.Linear(in_dim, hidden_dim)
        self.base_linear_2 = torch.nn.Linear(hidden_dim, 1)

        # Method-conditioned correction branch
        self.delta_linear_1 = torch.nn.Linear(in_dim, hidden_dim)
        self.delta_linear_2 = torch.nn.Linear(hidden_dim, 1)

        self.method_gamma = torch.nn.Linear(method_dim, hidden_dim)
        self.method_beta = torch.nn.Linear(method_dim, hidden_dim)

        # Identity FiLM initialization:
        # gamma(z)=0 and beta(z)=0, so h_delta' = h_delta initially.
        torch.nn.init.zeros_(self.method_gamma.weight)
        torch.nn.init.zeros_(self.method_gamma.bias)
        torch.nn.init.zeros_(self.method_beta.weight)
        torch.nn.init.zeros_(self.method_beta.bias)

        # Crucial: zero-initialize the final correction layer.
        # This makes delta_eps_i = 0 at initialization.
        torch.nn.init.zeros_(self.delta_linear_2.weight)
        torch.nn.init.zeros_(self.delta_linear_2.bias)

    def forward(
        self,
        x: torch.Tensor,
        method_z: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        if method_z is None:
            raise ValueError("DeltaReadoutFiLMBlock requires method_z.")
        if node_batch is None:
            raise ValueError("DeltaReadoutFiLMBlock requires node_batch.")

        # Base/shared readout
        h_base = self.base_linear_1(x)
        if self.gate is not None:
            h_base = self.gate(h_base)
        eps_base = self.base_linear_2(h_base)

        # Method-conditioned correction
        h_delta = self.delta_linear_1(x)
        if self.gate is not None:
            h_delta = self.gate(h_delta)

        z_nodes = method_z[node_batch].to(dtype=h_delta.dtype, device=h_delta.device)

        gamma = 1.0 + self.method_gamma(z_nodes)
        beta = self.method_beta(z_nodes)

        h_delta = gamma * h_delta + beta
        delta_eps = self.delta_linear_2(h_delta)

        return eps_base + delta_eps

# ResMLP readout block
@compile_mode("script")
class ReadoutResMLPBlock(torch.nn.Module):
    """Final scalar readout with direct ResMLP method conditioning.

    feature_mode="hidden":
        h_i = act(W_x x_i)
        eps_i = MLP([h_i, z_m])

    feature_mode="raw":
        eps_i = MLP([x_i, z_m])

    This block replaces the standard final nonlinear readout.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        method_dim: int,
        feature_mode: str = "hidden",
        cueq_config: Optional[Dict[str, Any]] = None,
        oeq_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        del cueq_config, oeq_config

        irreps_in = o3.Irreps(irreps_in)
        MLP_irreps = o3.Irreps(MLP_irreps)

        self.hidden_irreps = MLP_irreps

        if method_dim is None or method_dim <= 0:
            raise ValueError("ReadoutResMLPBlock requires method_dim > 0.")

        if feature_mode not in ("hidden", "raw"):
            raise ValueError(
                f"feature_mode must be 'hidden' or 'raw', got {feature_mode}."
            )

        if irreps_in.lmax > 0:
            raise ValueError(
                "ReadoutResMLPBlock expects scalar-only input irreps. "
                f"Got irreps_in={irreps_in}."
            )

        if MLP_irreps.lmax > 0:
            raise ValueError(
                "ReadoutResMLPBlock expects scalar-only MLP_irreps. "
                f"Got MLP_irreps={MLP_irreps}."
            )

        in_dim = irreps_in.count(o3.Irrep(0, 1))
        hidden_dim = MLP_irreps.count(o3.Irrep(0, 1))

        if in_dim <= 0:
            raise ValueError(f"Could not infer scalar input dim from {irreps_in}.")
        if hidden_dim <= 0:
            raise ValueError(f"Could not infer scalar hidden dim from {MLP_irreps}.")

        self.gate = gate
        self.feature_mode = feature_mode

        if feature_mode == "hidden":
            self.x_project = torch.nn.Linear(in_dim, hidden_dim)
            mlp_in_dim = hidden_dim + method_dim
        else:
            self.x_project = None
            mlp_in_dim = in_dim + method_dim

        self.cond_linear_1 = torch.nn.Linear(mlp_in_dim, hidden_dim)
        self.cond_linear_2 = torch.nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        method_z: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        if method_z is None:
            raise ValueError("ReadoutResMLPBlock requires method_z.")
        if node_batch is None:
            raise ValueError("ReadoutResMLPBlock requires node_batch.")

        z_nodes = method_z[node_batch].to(dtype=x.dtype, device=x.device)

        if self.feature_mode == "hidden":
            h = self.x_project(x)
            if self.gate is not None:
                h = self.gate(h)
            h = torch.cat([h, z_nodes], dim=-1)
        else:
            h = torch.cat([x, z_nodes], dim=-1)

        h = self.cond_linear_1(h)
        if self.gate is not None:
            h = self.gate(h)

        return self.cond_linear_2(h)


# Delta ResMLP readout block
@compile_mode("script")
class DeltaReadoutResMLPBlock(torch.nn.Module):
    """Shared base readout plus method-conditioned ResMLP correction.

    feature_mode="hidden":
        eps_i = eps_i_base(x_i) + MLP([h_i, z_m])
        h_i = act(W_delta x_i)

    feature_mode="raw":
        eps_i = eps_i_base(x_i) + MLP([x_i, z_m])

    The correction branch is zero-initialized, so initially:
        delta_eps_i = 0
    and the block behaves like a standard geometry-only readout.
    """

    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        method_dim: int,
        feature_mode: str = "hidden",
        cueq_config: Optional[Dict[str, Any]] = None,
        oeq_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        del cueq_config, oeq_config

        irreps_in = o3.Irreps(irreps_in)
        MLP_irreps = o3.Irreps(MLP_irreps)

        self.hidden_irreps = MLP_irreps

        if method_dim is None or method_dim <= 0:
            raise ValueError("DeltaReadoutResMLPBlock requires method_dim > 0.")

        if feature_mode not in ("hidden", "raw"):
            raise ValueError(
                f"feature_mode must be 'hidden' or 'raw', got {feature_mode}."
            )

        if irreps_in.lmax > 0:
            raise ValueError(
                "DeltaReadoutResMLPBlock expects scalar-only input irreps. "
                f"Got irreps_in={irreps_in}."
            )

        if MLP_irreps.lmax > 0:
            raise ValueError(
                "DeltaReadoutResMLPBlock expects scalar-only MLP_irreps. "
                f"Got MLP_irreps={MLP_irreps}."
            )

        in_dim = irreps_in.count(o3.Irrep(0, 1))
        hidden_dim = MLP_irreps.count(o3.Irrep(0, 1))

        if in_dim <= 0:
            raise ValueError(f"Could not infer scalar input dim from {irreps_in}.")
        if hidden_dim <= 0:
            raise ValueError(f"Could not infer scalar hidden dim from {MLP_irreps}.")

        self.gate = gate
        self.feature_mode = feature_mode

        # Base/shared readout branch: standard geometry-only readout
        self.base_linear_1 = torch.nn.Linear(in_dim, hidden_dim)
        self.base_linear_2 = torch.nn.Linear(hidden_dim, 1)

        # Method-conditioned residual branch
        if feature_mode == "hidden":
            self.delta_project = torch.nn.Linear(in_dim, hidden_dim)
            delta_in_dim = hidden_dim + method_dim
        else:
            self.delta_project = None
            delta_in_dim = in_dim + method_dim

        self.delta_linear_1 = torch.nn.Linear(delta_in_dim, hidden_dim)
        self.delta_linear_2 = torch.nn.Linear(hidden_dim, 1)

        # Important: start as a standard readout.
        torch.nn.init.zeros_(self.delta_linear_2.weight)
        torch.nn.init.zeros_(self.delta_linear_2.bias)

    def forward(
        self,
        x: torch.Tensor,
        method_z: torch.Tensor,
        node_batch: torch.Tensor,
    ) -> torch.Tensor:
        if method_z is None:
            raise ValueError("DeltaReadoutResMLPBlock requires method_z.")
        if node_batch is None:
            raise ValueError("DeltaReadoutResMLPBlock requires node_batch.")

        # Base/shared readout
        h_base = self.base_linear_1(x)
        if self.gate is not None:
            h_base = self.gate(h_base)
        eps_base = self.base_linear_2(h_base)

        # Method-conditioned correction
        z_nodes = method_z[node_batch].to(dtype=x.dtype, device=x.device)

        if self.feature_mode == "hidden":
            h_delta = self.delta_project(x)
            if self.gate is not None:
                h_delta = self.gate(h_delta)
            h_delta = torch.cat([h_delta, z_nodes], dim=-1)
        else:
            h_delta = torch.cat([x, z_nodes], dim=-1)

        h_delta = self.delta_linear_1(h_delta)
        if self.gate is not None:
            h_delta = self.gate(h_delta)

        delta_eps = self.delta_linear_2(h_delta)

        return eps_base + delta_eps

@simplify_if_compile
@compile_mode("script")
class NonLinearBiasReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Optional[Callable],
        irrep_out: o3.Irreps = o3.Irreps("0e"),
        num_heads: int = 1,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.num_heads = num_heads
        self.linear_1 = Linear(
            irreps_in=irreps_in, irreps_out=self.hidden_irreps, cueq_config=cueq_config
        )
        self.non_linearity = nn.Activation(irreps_in=self.hidden_irreps, acts=[gate])
        self.linear_mid = o3.Linear(
            irreps_in=self.hidden_irreps, irreps_out=self.hidden_irreps, biases=True
        )
        self.linear_2 = o3.Linear(
            irreps_in=self.hidden_irreps, irreps_out=irrep_out, biases=True
        )

    def forward(
        self, x: torch.Tensor, heads: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        if hasattr(self, "num_heads"):
            if self.num_heads > 1 and heads is not None:
                x = mask_head(x, heads, self.num_heads)
        x = self.non_linearity(self.linear_mid(x))
        if hasattr(self, "num_heads"):
            if self.num_heads > 1 and heads is not None:
                x = mask_head(x, heads, self.num_heads)
        return self.linear_2(x)  # [n_nodes, len(heads)]


@compile_mode("script")
class LinearDipoleReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        dipole_only: bool = False,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        if dipole_only:
            self.irreps_out = o3.Irreps("1x1o")
        else:
            self.irreps_out = o3.Irreps("1x0e + 1x1o")
        self.linear = Linear(
            irreps_in=irreps_in, irreps_out=self.irreps_out, cueq_config=cueq_config
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


@compile_mode("script")
class NonLinearDipoleReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Callable,
        dipole_only: bool = False,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        if dipole_only:
            self.irreps_out = o3.Irreps("1x1o")
        else:
            self.irreps_out = o3.Irreps("1x0e + 1x1o")
        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l == 0 and ir in self.irreps_out]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l > 0 and ir in self.irreps_out]
        )
        irreps_gates = o3.Irreps([mul, "0e"] for mul, _ in irreps_gated)
        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[gate for _, ir in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[gate] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()
        self.linear_1 = Linear(
            irreps_in=irreps_in, irreps_out=self.irreps_nonlin, cueq_config=cueq_config
        )
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irreps_out,
            cueq_config=cueq_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.equivariant_nonlin(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]


@compile_mode("script")
class LinearDipolePolarReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        use_polarizability: bool = True,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        if use_polarizability:
            print("You will calculate the polarizability and dipole.")
            self.irreps_out = o3.Irreps("2x0e + 1x1o + 1x2e")
        else:
            raise ValueError(
                "Invalid configuration for LinearDipolePolarReadoutBlock: "
                "use_polarizability must be either True."
                "If you want to calculate only the dipole, use AtomicDipolesMACE."
            )

        self.linear = Linear(
            irreps_in=irreps_in, irreps_out=self.irreps_out, cueq_config=cueq_config
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        y = self.linear(x)  # [n_nodes, 1]
        return y  # [n_nodes, 1]


@compile_mode("script")
class NonLinearDipolePolarReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Callable,
        use_polarizability: bool = True,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        if use_polarizability:
            print("You will calculate the polarizability and dipole.")
            self.irreps_out = o3.Irreps("2x0e + 1x1o + 1x2e")
        else:
            raise ValueError(
                "Invalid configuration for NonLinearDipolePolarReadoutBlock: "
                "use_polarizability must be either True."
                "If you want to calculate only the dipole, use AtomicDipolesMACE."
            )
        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l == 0 and ir in self.irreps_out]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l > 0 and ir in self.irreps_out]
        )
        irreps_gates = o3.Irreps([mul, "0e"] for mul, _ in irreps_gated)
        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[gate for _, ir in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[gate] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()
        self.linear_1 = Linear(
            irreps_in=irreps_in, irreps_out=self.irreps_nonlin, cueq_config=cueq_config
        )
        self.linear_2 = Linear(
            irreps_in=self.hidden_irreps,
            irreps_out=self.irreps_out,
            cueq_config=cueq_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.equivariant_nonlin(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]


@compile_mode("script")
class AtomicEnergiesBlock(torch.nn.Module):
    atomic_energies: torch.Tensor

    def __init__(self, atomic_energies: Union[np.ndarray, torch.Tensor]):
        super().__init__()
        # assert len(atomic_energies.shape) == 1

        self.register_buffer(
            "atomic_energies",
            torch.tensor(atomic_energies, dtype=torch.get_default_dtype()),
        )  # [n_elements, n_heads]

    def forward(
        self, x: torch.Tensor  # one-hot of elements [..., n_elements]
    ) -> torch.Tensor:  # [..., ]
        return torch.matmul(x, torch.atleast_2d(self.atomic_energies).T)

    def __repr__(self):
        formatted_energies = ", ".join(
            [
                "[" + ", ".join([f"{x:.4f}" for x in group]) + "]"
                for group in torch.atleast_2d(self.atomic_energies)
            ]
        )
        return f"{self.__class__.__name__}(energies=[{formatted_energies}])"


@compile_mode("script")
class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        radial_type: str = "bessel",
        distance_transform: str = "None",
        apply_cutoff: bool = True,
    ):
        super().__init__()
        if radial_type == "bessel":
            self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_type == "gaussian":
            self.bessel_fn = GaussianBasis(r_max=r_max, num_basis=num_bessel)
        elif radial_type == "chebyshev":
            self.bessel_fn = ChebychevBasis(r_max=r_max, num_basis=num_bessel)
        if distance_transform == "Agnesi":
            self.distance_transform = AgnesiTransform()
        elif distance_transform == "Soft":
            self.distance_transform = SoftTransform()
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel
        self.apply_cutoff = apply_cutoff

    def forward(
        self,
        edge_lengths: torch.Tensor,  # [n_edges, 1]
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ):
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        if hasattr(self, "distance_transform"):
            edge_lengths = self.distance_transform(
                edge_lengths, node_attrs, edge_index, atomic_numbers
            )
        radial = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        if hasattr(self, "apply_cutoff"):
            if not self.apply_cutoff:
                return radial, cutoff
        return radial * cutoff, None  # [n_edges, n_basis], [n_edges, 1]


@compile_mode("script")
class EquivariantProductBasisBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        correlation: int,
        use_sc: bool = True,
        num_elements: Optional[int] = None,
        use_agnostic_product: bool = False,
        use_reduced_cg: Optional[bool] = None,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,
    ) -> None:
        super().__init__()

        self.use_sc = use_sc
        self.use_agnostic_product = use_agnostic_product
        if self.use_agnostic_product:
            num_elements = 1
        self.symmetric_contractions = SymmetricContractionWrapper(
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            num_elements=num_elements,
            use_reduced_cg=use_reduced_cg,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
        )
        # Update linear
        self.linear = Linear(
            target_irreps,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=cueq_config,
        )
        self.cueq_config = cueq_config

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: Optional[torch.Tensor],
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        use_cueq = False
        use_cueq_mul_ir = False
        if hasattr(self, "use_agnostic_product"):
            if self.use_agnostic_product:
                node_attrs = torch.ones(
                    (node_feats.shape[0], 1),
                    dtype=node_feats.dtype,
                    device=node_feats.device,
                )
        if hasattr(self, "cueq_config"):
            if self.cueq_config is not None:
                if self.cueq_config.enabled and (
                    self.cueq_config.optimize_all or self.cueq_config.optimize_symmetric
                ):
                    use_cueq = True
                if self.cueq_config.layout_str == "mul_ir":
                    use_cueq_mul_ir = True
        if use_cueq:
            if use_cueq_mul_ir:
                node_feats = torch.transpose(node_feats, 1, 2)
            index_attrs = torch.nonzero(node_attrs)[:, 1].int()
            node_feats = self.symmetric_contractions(
                node_feats.flatten(1),
                index_attrs,
            )
        else:
            node_feats = self.symmetric_contractions(node_feats, node_attrs)
        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc
        return self.linear(node_feats)

## add conditioned MLP module
@compile_mode("script")
class ConcatConditionedRadialMLP(torch.nn.Module):
    def __init__(
        self,
        edge_input_dim: int,
        method_dim: int,
        hidden_dims: List[int],
        out_dim: int,
    ):
        super().__init__()
        self.edge_input_dim = edge_input_dim
        self.method_dim = method_dim
        self.hidden_dims = list(hidden_dims)

        layers: List[torch.nn.Module] = []
        in_dim = edge_input_dim + method_dim
        for h_dim in hidden_dims:
            layers.append(torch.nn.Linear(in_dim, h_dim))
            layers.append(torch.nn.SiLU())
            in_dim = h_dim
        layers.append(torch.nn.Linear(in_dim, out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(
        self,
        edge_feats: torch.Tensor,
        method_z: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([edge_feats, method_z.to(edge_feats.dtype)], dim=-1)
        return self.net(x)

# new film radial conditioning
@compile_mode("script")
class FiLMConditionedRadialMLP(torch.nn.Module):
    def __init__(
        self,
        edge_input_dim: int,
        method_dim: int,
        hidden_dims: List[int],
        out_dim: int,
    ):
        super().__init__()
        self.edge_input_dim = edge_input_dim
        self.method_dim = method_dim
        self.hidden_dims = list(hidden_dims)
        self.out_dim = out_dim

        self.hidden_linears = torch.nn.ModuleList()
        self.film_gamma = torch.nn.ModuleList()
        self.film_beta = torch.nn.ModuleList()

        in_dim = edge_input_dim
        for h_dim in hidden_dims:
            self.hidden_linears.append(torch.nn.Linear(in_dim, h_dim))
            self.film_gamma.append(torch.nn.Linear(method_dim, h_dim))
            self.film_beta.append(torch.nn.Linear(method_dim, h_dim))
            in_dim = h_dim

        self.out_linear = torch.nn.Linear(in_dim, out_dim)
        self.act = torch.nn.SiLU()

        # identity init for FiLM: gamma(z)=0, beta(z)=0
        for gamma_layer, beta_layer in zip(self.film_gamma, self.film_beta):
            torch.nn.init.zeros_(gamma_layer.weight)
            torch.nn.init.zeros_(gamma_layer.bias)
            torch.nn.init.zeros_(beta_layer.weight)
            torch.nn.init.zeros_(beta_layer.bias)

    def forward(
        self,
        edge_feats: torch.Tensor,
        method_z: torch.Tensor,
    ) -> torch.Tensor:
        x = edge_feats
        z = method_z.to(edge_feats.dtype)

        for linear, gamma_layer, beta_layer in zip(
            self.hidden_linears, self.film_gamma, self.film_beta
        ):
            x = linear(x)
            x = self.act(x)
            gamma = 1.0 + gamma_layer(z)
            beta = beta_layer(z)
            x = gamma * x + beta

        return self.out_linear(x)

@compile_mode("script")
class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        edge_irreps: Optional[o3.Irreps] = None,
        radial_MLP: Optional[List[int]] = None,
        interaction_method: str = "none",
        method_emb_dim: int = 0,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        if edge_irreps is None:
            edge_irreps = self.node_feats_irreps
        self.radial_MLP = radial_MLP
        self.edge_irreps = edge_irreps
        self.interaction_method = interaction_method
        self.method_emb_dim = method_emb_dim
        self.cueq_config = cueq_config
        self.oeq_config = oeq_config
        if self.oeq_config and self.oeq_config.conv_fusion:
            self.conv_fusion = self.oeq_config.conv_fusion
        if self.cueq_config and self.cueq_config.conv_fusion:
            self.conv_fusion = self.cueq_config.conv_fusion
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    def handle_lammps(
        self,
        node_feats: torch.Tensor,
        lammps_class: Optional[Any],
        lammps_natoms: Tuple[int, int],
        first_layer: bool,
    ) -> torch.Tensor:  # noqa: D401 – internal helper
        if lammps_class is None or first_layer or torch.jit.is_scripting():
            return node_feats
        _, n_total = lammps_natoms
        pad = torch.zeros(
            (n_total, node_feats.shape[1]),
            dtype=node_feats.dtype,
            device=node_feats.device,
        )
        node_feats = torch.cat((node_feats, pad), dim=0)
        node_feats = LAMMPS_MP.apply(node_feats, lammps_class)
        return node_feats

    def truncate_ghosts(
        self, tensor: torch.Tensor, n_real: Optional[int] = None
    ) -> torch.Tensor:
        """Truncate the tensor to only keep the real atoms in case of presence of ghost atoms during multi-GPU MD simulations."""
        return tensor[:n_real] if n_real is not None else tensor

    ## add helper methods for radial conditioning
    def _make_conv_tp_weights(
        self,
        input_dim: int,
        out_dim: int,
    ) -> torch.nn.Module:
        if self.interaction_method == "none":
            return nn.FullyConnectedNet(
                [input_dim] + self.radial_MLP + [out_dim],
                torch.nn.functional.silu,
            )

        if self.interaction_method == "radial_concat":
            if self.method_emb_dim <= 0:
                raise ValueError(
                    "interaction_method='radial_concat' requires method_emb_dim > 0"
                )
            return ConcatConditionedRadialMLP(
                edge_input_dim=input_dim,
                method_dim=self.method_emb_dim,
                hidden_dims=self.radial_MLP,
                out_dim=out_dim,
            )

        if self.interaction_method == "radial_film":
            if self.method_emb_dim <= 0:
                raise ValueError(
                    "interaction_method='radial_film' requires method_emb_dim > 0"
                )
            return FiLMConditionedRadialMLP(
                edge_input_dim=input_dim,
                method_dim=self.method_emb_dim,
                hidden_dims=self.radial_MLP,
                out_dim=out_dim,
            )

        raise ValueError(f"Unknown interaction_method: {self.interaction_method}")

    def _compute_conv_tp_weights(
        self,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        method_z: Optional[torch.Tensor] = None,
        node_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.interaction_method == "none":
            return self.conv_tp_weights(edge_feats)

        if self.interaction_method in ("radial_concat", "radial_film"):
            if method_z is None:
                raise ValueError(
                    f"interaction_method='{self.interaction_method}' requires method_z in forward()"
                )
            if node_batch is None:
                raise ValueError(
                    f"interaction_method='{self.interaction_method}' requires node_batch in forward()"
                )
            edge_batch = node_batch[edge_index[0]]
            z_edge = method_z[edge_batch]
            return self.conv_tp_weights(edge_feats, z_edge)

        raise ValueError(f"Unknown interaction_method: {self.interaction_method}")

    @abstractmethod
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
        method_z: Optional[torch.Tensor] = None,
        node_batch: Optional[torch.Tensor] = None,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError


nonlinearities = {1: torch.nn.functional.silu, -1: torch.tanh}


@compile_mode("script")
class RealAgnosticInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        if not hasattr(self, "oeq_config"):
            self.oeq_config = None

        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.edge_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.edge_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
            oeq_config=self.oeq_config,
        )

        # Convolution weights
        # original
        #input_dim = self.edge_feats_irreps.num_irreps
        #self.conv_tp_weights = nn.FullyConnectedNet(
        #    [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
        #    torch.nn.functional.silu,
        #)
        # new with radial MLP conditioning
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = self._make_conv_tp_weights(
            input_dim=input_dim,
            out_dim=self.conv_tp.weight_numel,
        )        

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
        method_z: Optional[torch.Tensor] = None,
        node_batch: Optional[torch.Tensor] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        lammps_class: Optional[Any] = None,
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        n_real = lammps_natoms[0] if lammps_class is not None else None
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        # before
        #tp_weights = self.conv_tp_weights(edge_feats)
        # with radial MLP conditioning
        tp_weights = self._compute_conv_tp_weights(
            edge_feats=edge_feats,
            edge_index=edge_index,
            method_z=method_z,
            node_batch=node_batch,
        )

        if cutoff is not None:
            tp_weights = tp_weights * cutoff

        message = None
        if hasattr(self, "conv_fusion"):
            message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
        else:
            mji = self.conv_tp(
                node_feats[edge_index[0]], edge_attrs, tp_weights
            )  # [n_nodes, irreps]
            message = scatter_sum(
                src=mji, index=edge_index[1], dim=0, dim_size=node_feats.shape[0]
            )
        message = self.truncate_ghosts(message, n_real)
        node_attrs = self.truncate_ghosts(node_attrs, n_real)
        message = self.linear(message) / self.avg_num_neighbors
        message = self.skip_tp(message, node_attrs)
        return (
            self.reshape(message),
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        if not hasattr(self, "oeq_config"):
            self.oeq_config = None

        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.edge_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.edge_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
            oeq_config=self.oeq_config,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        #self.conv_tp_weights = nn.FullyConnectedNet(
        #    [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
        #    torch.nn.functional.silu,  # gate
        #)
        # with radial MLP conditioning
        self.conv_tp_weights = self._make_conv_tp_weights(
                    input_dim=input_dim,
                    out_dim=self.conv_tp.weight_numel,
                )        

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
        )
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
        method_z: Optional[torch.Tensor] = None,
        node_batch: Optional[torch.Tensor] = None,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_real = lammps_natoms[0] if lammps_class is not None else None
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        #tp_weights = self.conv_tp_weights(edge_feats)
        tp_weights = self._compute_conv_tp_weights(
            edge_feats=edge_feats,
            edge_index=edge_index,
            method_z=method_z,
            node_batch=node_batch,
        )

        if cutoff is not None:
            tp_weights = tp_weights * cutoff
        message = None
        if hasattr(self, "conv_fusion"):
            message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
        else:
            mji = self.conv_tp(
                node_feats[edge_index[0]], edge_attrs, tp_weights
            )  # [n_nodes, irreps]
            message = scatter_sum(
                src=mji, index=edge_index[1], dim=0, dim_size=node_feats.shape[0]
            )
        message = self.truncate_ghosts(message, n_real)
        node_attrs = self.truncate_ghosts(node_attrs, n_real)
        sc = self.truncate_ghosts(sc, n_real)
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticDensityInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        if not hasattr(self, "oeq_config"):
            self.oeq_config = None

        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.edge_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.edge_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
            oeq_config=self.oeq_config,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        #self.conv_tp_weights = nn.FullyConnectedNet(
        #    [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
        #    torch.nn.functional.silu,
        #)
        # with radial MLP conditioning
        self.conv_tp_weights = self._make_conv_tp_weights(
            input_dim=input_dim,
            out_dim=self.conv_tp.weight_numel,
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.irreps_out,
            self.node_attrs_irreps,
            self.irreps_out,
            cueq_config=self.cueq_config,
        )

        # Density normalization
        self.density_fn = nn.FullyConnectedNet(
            [input_dim]
            + [
                1,
            ],
            torch.nn.functional.silu,
        )
        # Reshape
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
        method_z: Optional[torch.Tensor] = None,
        node_batch: Optional[torch.Tensor] = None,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        n_real = lammps_natoms[0] if lammps_class is not None else None
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        #tp_weights = self.conv_tp_weights(edge_feats)
        tp_weights = self._compute_conv_tp_weights(
            edge_feats=edge_feats,
            edge_index=edge_index,
            method_z=method_z,
            node_batch=node_batch,
        )
        
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff
            edge_density = edge_density * cutoff
        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]
        message = None
        if hasattr(self, "conv_fusion"):
            message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
        else:
            mji = self.conv_tp(
                node_feats[edge_index[0]], edge_attrs, tp_weights
            )  # [n_nodes, irreps]
            message = scatter_sum(
                src=mji, index=edge_index[1], dim=0, dim_size=node_feats.shape[0]
            )

        message = self.truncate_ghosts(message, n_real)
        node_attrs = self.truncate_ghosts(node_attrs, n_real)
        density = self.truncate_ghosts(density, n_real)
        message = self.linear(message) / (density + 1)
        message = self.skip_tp(message, node_attrs)
        return (
            self.reshape(message),
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticDensityResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        if not hasattr(self, "oeq_config"):
            self.oeq_config = None

        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.edge_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.edge_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
            oeq_config=self.oeq_config,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        #self.conv_tp_weights = nn.FullyConnectedNet(
        #    [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
        #    torch.nn.functional.silu,  # gate
        #)
        # with radial MLP conditioning
        self.conv_tp_weights = self._make_conv_tp_weights(
            input_dim=input_dim,
            out_dim=self.conv_tp.weight_numel,
        )

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Selector TensorProduct
        self.skip_tp = FullyConnectedTensorProduct(
            self.node_feats_irreps,
            self.node_attrs_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
        )

        # Density normalization
        self.density_fn = nn.FullyConnectedNet(
            [input_dim]
            + [
                1,
            ],
            torch.nn.functional.silu,
        )

        # Reshape
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
        method_z: Optional[torch.Tensor] = None,
        node_batch: Optional[torch.Tensor] = None,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        n_real = lammps_natoms[0] if lammps_class is not None else None
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )
        #tp_weights = self.conv_tp_weights(edge_feats)
        tp_weights = self._compute_conv_tp_weights(
            edge_feats=edge_feats,
            edge_index=edge_index,
            method_z=method_z,
            node_batch=node_batch,
        )
        edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff
            edge_density = edge_density * cutoff
        density = scatter_sum(
            src=edge_density, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, 1]

        message = None
        if hasattr(self, "conv_fusion"):
            message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
        else:
            mji = self.conv_tp(
                node_feats[edge_index[0]], edge_attrs, tp_weights
            )  # [n_nodes, irreps]
            message = scatter_sum(
                src=mji, index=edge_index[1], dim=0, dim_size=node_feats.shape[0]
            )

        message = self.truncate_ghosts(message, n_real)
        node_attrs = self.truncate_ghosts(node_attrs, n_real)
        density = self.truncate_ghosts(density, n_real)
        sc = self.truncate_ghosts(sc, n_real)
        message = self.linear(message) / (density + 1)
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticAttResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        if not hasattr(self, "oeq_config"):
            self.oeq_config = None

        self.node_feats_down_irreps = o3.Irreps("64x0e")
        # First linear
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.edge_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.edge_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
            oeq_config=self.oeq_config,
        )

        # Convolution weights
        # conditioned radial MLP with node feature concatenation
        self.linear_down = Linear(
            self.node_feats_irreps,
            self.node_feats_down_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        input_dim = (
            self.edge_feats_irreps.num_irreps
            + 2 * self.node_feats_down_irreps.num_irreps
        )

        # interaction method options
        if self.interaction_method == "none":
            self.conv_tp_weights = nn.FullyConnectedNet(
                [input_dim] + 3 * [256] + [self.conv_tp.weight_numel],
                torch.nn.functional.silu,
            )
        elif self.interaction_method == "radial_concat":
            if self.method_emb_dim <= 0:
                raise ValueError(
                    "interaction_method='radial_concat' requires method_emb_dim > 0"
                )
            self.conv_tp_weights = ConcatConditionedRadialMLP(
                edge_input_dim=input_dim,
                method_dim=self.method_emb_dim,
                hidden_dims=[256, 256, 256],
                out_dim=self.conv_tp.weight_numel,
            )
        elif self.interaction_method == "radial_film":
            if self.method_emb_dim <= 0:
                raise ValueError(
                    "interaction_method='radial_film' requires method_emb_dim > 0"
                )
            self.conv_tp_weights = FiLMConditionedRadialMLP(
                edge_input_dim=input_dim,
                method_dim=self.method_emb_dim,
                hidden_dims=[256, 256, 256],
                out_dim=self.conv_tp.weight_numel,
            )
        else:
            raise ValueError(f"Unknown interaction_method: {self.interaction_method}")

        # Linear
        self.irreps_out = self.target_irreps
        self.linear = Linear(
            irreps_mid,
            self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

        # Skip connection.
        self.skip_linear = Linear(
            self.node_feats_irreps, self.hidden_irreps, cueq_config=self.cueq_config
        )

    # pylint: disable=unused-argument
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
        method_z: Optional[torch.Tensor] = None,
        node_batch: Optional[torch.Tensor] = None,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        sc = self.skip_linear(node_feats)
        node_feats_up = self.linear_up(node_feats)
        node_feats_down = self.linear_down(node_feats)
        augmented_edge_feats = torch.cat(
            [
                edge_feats,
                node_feats_down[sender],
                node_feats_down[receiver],
            ],
            dim=-1,
        )
        #tp_weights = self.conv_tp_weights(augmented_edge_feats)
        # with radial MLP conditioning
        tp_weights = self._compute_conv_tp_weights(
            edge_feats=augmented_edge_feats,
            edge_index=edge_index,
            method_z=method_z,
            node_batch=node_batch,
        )

        if cutoff is not None:
            tp_weights = tp_weights * cutoff
        message = None
        if hasattr(self, "conv_fusion"):
            message = self.conv_tp(node_feats_up, edge_attrs, tp_weights, edge_index)
        else:
            mji = self.conv_tp(
                node_feats_up[edge_index[0]], edge_attrs, tp_weights
            )  # [n_nodes, irreps]
            message = scatter_sum(
                src=mji, index=edge_index[1], dim=0, dim_size=node_feats.shape[0]
            )
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticResidualNonLinearInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        if not hasattr(self, "cueq_config"):
            self.cueq_config = None
        # First linear
        node_scalar_irreps = o3.Irreps(
            [(self.node_feats_irreps.count(o3.Irrep(0, 1)), (0, 1))]
        )
        self.source_embedding = Linear(
            self.node_attrs_irreps,
            node_scalar_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        self.target_embedding = Linear(
            self.node_attrs_irreps,
            node_scalar_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        self.linear_up = Linear(
            self.node_feats_irreps,
            self.edge_irreps,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        torch.nn.init.uniform_(self.source_embedding.weight, a=-0.001, b=0.001)
        torch.nn.init.uniform_(self.target_embedding.weight, a=-0.001, b=0.001)

        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.edge_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = TensorProduct(
            self.edge_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
            cueq_config=self.cueq_config,
        )

        # Convolution weights
        #input_dim = self.edge_feats_irreps.num_irreps
        #self.conv_tp_weights = RadialMLP(
        #    [input_dim + 2 * node_scalar_irreps.dim]
        #    + self.radial_MLP
        #    + [self.conv_tp.weight_numel]
        #)
        #self.irreps_out = self.target_irreps
        # Convolution weights
        # with radial MLP conditioning on edge features and node scalar embeddings
        input_dim = self.edge_feats_irreps.num_irreps + 2 * node_scalar_irreps.dim

        if self.interaction_method == "none":
            self.conv_tp_weights = RadialMLP(
                [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel]
            )
        elif self.interaction_method == "radial_concat":
            if self.method_emb_dim <= 0:
                raise ValueError(
                    "interaction_method='radial_concat' requires method_emb_dim > 0"
                )
            self.conv_tp_weights = ConcatConditionedRadialMLP(
                edge_input_dim=input_dim,
                method_dim=self.method_emb_dim,
                hidden_dims=self.radial_MLP,
                out_dim=self.conv_tp.weight_numel,
            )
        elif self.interaction_method == "radial_film":
            if self.method_emb_dim <= 0:
                raise ValueError(
                    "interaction_method='radial_film' requires method_emb_dim > 0"
                )
            self.conv_tp_weights = FiLMConditionedRadialMLP(
                edge_input_dim=input_dim,
                method_dim=self.method_emb_dim,
                hidden_dims=self.radial_MLP,
                out_dim=self.conv_tp.weight_numel,
            )
        else:
            raise ValueError(f"Unknown interaction_method: {self.interaction_method}")

        self.irreps_out = self.target_irreps


        # Selector TensorProduct
        self.skip_tp = Linear(
            self.node_feats_irreps,
            self.hidden_irreps,
            cueq_config=self.cueq_config,
        )
        self.reshape = reshape_irreps(self.irreps_out, cueq_config=self.cueq_config)

        # Non-linearity
        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in self.irreps_out if ir.l == 0]
        )
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l > 0])
        irreps_gates = o3.Irreps([mul, "0e"] for mul, _ in irreps_gated)
        activation_fn = torch.nn.functional.silu
        act_gates_fn = torch.nn.functional.sigmoid
        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[activation_fn for _ in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[act_gates_fn] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()

        # Linear residual
        self.linear_res = Linear(
            self.edge_irreps,
            self.irreps_nonlin,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Linear
        self.linear_1 = Linear(
            irreps_mid,
            self.irreps_nonlin,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )
        self.linear_2 = Linear(
            irreps_in=self.irreps_out,
            irreps_out=self.irreps_out,
            internal_weights=True,
            shared_weights=True,
            cueq_config=self.cueq_config,
        )

        # Normalizations
        #self.density_fn = RadialMLP(
        #    [input_dim + 2 * node_scalar_irreps.dim] + [64] + [1],
        #)
        self.density_fn = RadialMLP(
            [input_dim] + [64] + [1],
        )

        self.alpha = torch.nn.Parameter(torch.tensor(20.0), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

        self.transpose_mul_ir = TransposeIrrepsLayoutWrapper(
            irreps=self.irreps_nonlin,
            source="ir_mul",
            target="mul_ir",
            cueq_config=self.cueq_config,
        )
        self.transpose_ir_mul = TransposeIrrepsLayoutWrapper(
            irreps=self.irreps_out,
            source="mul_ir",
            target="ir_mul",
            cueq_config=self.cueq_config,
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
        cutoff: Optional[torch.Tensor] = None,
        method_z: Optional[torch.Tensor] = None,
        node_batch: Optional[torch.Tensor] = None,
        lammps_class: Optional[Any] = None,
        lammps_natoms: Tuple[int, int] = (0, 0),
        first_layer: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = node_feats.shape[0]
        n_real = lammps_natoms[0] if lammps_class is not None else None
        sc = self.skip_tp(node_feats)
        node_feats = self.linear_up(node_feats)
        node_feats_res = self.linear_res(node_feats)
        node_feats = self.handle_lammps(
            node_feats,
            lammps_class=lammps_class,
            lammps_natoms=lammps_natoms,
            first_layer=first_layer,
        )

        source_embedding = self.source_embedding(node_attrs)
        target_embedding = self.target_embedding(node_attrs)
        # before
        #edge_feats = torch.cat(
        #    [
        #        edge_feats,
        #        source_embedding[edge_index[0]],
        #        target_embedding[edge_index[1]],
        #    ],
        #    dim=-1,
        #)
        #tp_weights = self.conv_tp_weights(edge_feats)

        #edge_density = torch.tanh(self.density_fn(edge_feats) ** 2)
        # with radial MLP conditioning on edge features and node scalar embeddings
        aug_edge_feats = torch.cat(
            [
                edge_feats,
                source_embedding[edge_index[0]],
                target_embedding[edge_index[1]],
            ],
            dim=-1,
        )

        tp_weights = self._compute_conv_tp_weights(
            edge_feats=aug_edge_feats,
            edge_index=edge_index,
            method_z=method_z,
            node_batch=node_batch,
        )

        edge_density = torch.tanh(self.density_fn(aug_edge_feats) ** 2)
        
        
        if cutoff is not None:
            tp_weights = tp_weights * cutoff
            edge_density = edge_density * cutoff
        density = scatter_sum(
            src=edge_density, index=edge_index[1], dim=0, dim_size=num_nodes
        )

        if hasattr(self, "conv_fusion"):
            message = self.conv_tp(node_feats, edge_attrs, tp_weights, edge_index)
        else:
            mji = self.conv_tp(
                node_feats[edge_index[0]], edge_attrs, tp_weights
            )  # [n_edges, irreps]
            message = scatter_sum(
                src=mji, index=edge_index[1], dim=0, dim_size=num_nodes
            )  # [n_nodes, irreps]

        message = self.truncate_ghosts(message, n_real)
        density = self.truncate_ghosts(density, n_real)
        sc = self.truncate_ghosts(sc, n_real)
        node_feats_res = self.truncate_ghosts(node_feats_res, n_real)
        message = self.linear_1(message) / (density * self.beta + self.alpha)
        message = message + node_feats_res
        if self.transpose_mul_ir is not None:
            message = self.transpose_mul_ir(message)
        message = self.equivariant_nonlin(message)
        if self.transpose_ir_mul is not None:
            message = self.transpose_ir_mul(message)
        message = self.linear_2(message)
        return (
            self.reshape(message),
            sc,
        )


@compile_mode("script")
class ScaleShiftBlock(torch.nn.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.register_buffer(
            "scale",
            torch.tensor(scale, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "shift",
            torch.tensor(shift, dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
        return (
            torch.atleast_1d(self.scale)[head] * x + torch.atleast_1d(self.shift)[head]
        )

    def __repr__(self):
        formatted_scale = (
            ", ".join([f"{x:.4f}" for x in self.scale])
            if self.scale.numel() > 1
            else f"{self.scale.item():.4f}"
        )
        formatted_shift = (
            ", ".join([f"{x:.4f}" for x in self.shift])
            if self.shift.numel() > 1
            else f"{self.shift.item():.4f}"
        )
        return f"{self.__class__.__name__}(scale={formatted_scale}, shift={formatted_shift})"
