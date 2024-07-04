"""Poincare ball manifold."""
import sys
sys.path.append('/data/lige/HKN')# Please change accordingly!

import torch

from manifolds.base import Manifold
from manifolds.base import ManifoldParameter
from utils.math_utils import artanh, tanh


class PoincareBall(Manifold):
    """
    PoicareBall Manifold class.

    We use the following convention: x0^2 + x1^2 + ... + xd^2 < 1 / c

    Note that 1/sqrt(c) is the Poincare ball radius.

    """

    def __init__(self, ):
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-15
        self.max_artanh = 1 - 1e-5
        self.max_tanh = 15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def sqdist(self, p1, p2, c):
        sqrt_c = c ** 0.5
        """
        dist_c = artanh(
            sqrt_c * self.mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
        )
        """
        mobius_sum = self.mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
        clipped_sum = torch.clamp(sqrt_c * mobius_sum, max=self.max_artanh)
        dist_c = artanh(clipped_sum)
        dist = dist_c * 2 / (sqrt_c.clamp_min(self.eps[p1.dtype]))
        return dist ** 2

    def _lambda_x(self, x, c):
        x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
        return 2 / ((1. - c * x_sqnorm).clamp_min(self.eps[x.dtype]))

    def egrad2rgrad(self, p, dp, c):
        lambda_p = self._lambda_x(p, c)
        dp /= lambda_p.pow(2)
        return dp

    #This ensures hyperbolic vectors are not too close or even beyond the border, (HNN paper)
    #This avoids many numerical errors
    def proj(self, x, c):
        norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), self.min_norm) #perturb from o
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x) #not too close to border

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        second_term = (
                tanh((sqrt_c / 2 * self._lambda_x(p, c) * u_norm).clamp(-self.max_tanh, self.max_tanh))
                * u
                / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term, c)
        return self.proj(gamma_1, c)

    def logmap(self, p1, p2, c):
        sub = self.mobius_add(-p1, p2, c)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        lam = self._lambda_x(p1, c)
        sqrt_c = c ** 0.5
        return 2 / sqrt_c / lam * artanh((sqrt_c * sub_norm).clamp(-self.max_artanh, self.max_artanh)) * sub / sub_norm

    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), self.min_norm)
        gamma_1 = tanh((sqrt_c * u_norm).clamp(-self.max_tanh, self.max_tanh)) * u / (sqrt_c * u_norm)
        return self.proj(gamma_1, c)

    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = 1. / sqrt_c * artanh((sqrt_c * p_norm).clamp(-self.max_artanh, self.max_artanh)) / p_norm
        return scale * p

    def mobius_add(self, x, y, c, dim=-1):
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / (denom.clamp_min(self.eps[x.dtype]))

    def mobius_matvec(self, m, x, c):
        sqrt_c = c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.transpose(-1, -2)
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = tanh((mx_norm / x_norm * artanh((sqrt_c * x_norm).clamp(-self.max_artanh, self.max_artanh))).clamp(-self.max_tanh, self.max_tanh)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        res = torch.where(cond.bool(), res_c, res_0) #torch.where(cond, res_0, res_c)
        return res

    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def _gyration(self, u, v, w, c, dim: int = -1):
        u2 = u.pow(2).sum(dim=dim, keepdim=True)
        v2 = v.pow(2).sum(dim=dim, keepdim=True)
        uv = (u * v).sum(dim=dim, keepdim=True)
        uw = (u * w).sum(dim=dim, keepdim=True)
        vw = (v * w).sum(dim=dim, keepdim=True)
        c2 = c ** 2
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        return w + 2 * (a * u + b * v) / d.clamp_min(self.min_norm)

    def inner(self, x, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        lambda_x = self._lambda_x(x, c)
        return lambda_x ** 2 * (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    def ptransp_(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        return self._gyration(y, -x, u, c) * lambda_x / lambda_y

    #PT a vector u from the origin o to x
    def ptransp0(self, x, u, c):
        lambda_x = self._lambda_x(x, c)
        return 2 * u / lambda_x.clamp_min(self.min_norm)

#################################Additional functions implemented!#################################
    def HCDist(self, x, cls,c):
        """
        Calculate Poincare distances between points x and "class-points" cls without using loops.

        Parameters:
        x: Tensor of shape (n, d') representing the points.
        cls: Tensor of shape (num_classes, d') representing the class centers.

        Returns:
        Euclidean_x: Tensor of shape (n, num_classes) representing the distances.

        Remarks:
        The problem with sqdist is that it takes 2 tensors of the same shape, and calculate "all distance pairs"
        for the very last dimension: (n,K,nei_num,d')*(n,K,nei_num,d')->(n,K,nei_num)
        Nevertheless, when decoding from Hyperbolic representations to Euclidean representations,
        x:(n,d'),cls:(num_classes,d'), the input has different shapes, and we only cares about certain
        pairs of distances. So we need a proper distance calculation functions here.
        """
        x_expanded = x.unsqueeze(1)  # Shape: (n, 1, d')
        cls_expanded = cls.unsqueeze(0)  # Shape: (1, num_classes, d')
        Euclidean_x = torch.sqrt(self.sqdist(x_expanded, cls_expanded,c=c)).clamp_min(self.min_norm)  # Shape: (n, num_classes)

        return Euclidean_x
    
    #PT a vector u from the x to origin o
    def ptransp0back(self, x, u, c):
        lambda_x = self._lambda_x(x, c)
        
        return (lambda_x/2*u).clamp_min(self.min_norm)

    def origin(self, *size, c, dtype=None, device=None) -> "ManifoldParameter":
        if dtype is None:
            dtype = c.dtype
        if device is None:
            device = c.device

        zero_point = torch.zeros(*size, dtype=dtype, device=device)
        return ManifoldParameter(zero_point, manifold=self,requires_grad=False,c=c)

    #This is somewhat wrong, tensor on diff device issue, not sure how to fix
    def random_normal(self, size, c, mean=0, std=1, dtype=None, device=None) -> "ManifoldParameter":
        if dtype is None:
            dtype = torch.float64
        if device is None:
            device = torch.device('cpu')

        tangents = torch.randn(size, dtype=dtype, device=device) * std + mean
        #print("tangents device: ",tangents.device)cpu
        tangents /= tangents.norm(dim=-1, keepdim=True)
        #print("tangents device: ",tangents.device) cpu

        return self.expmap0(tangents,c=c.to(device))                              

    
    def poincare_to_klein(self, x, c):
        norm_x = torch.norm(x, dim=-1, keepdim=True)#Because point feature always lie in the last dimension
        factor = 2 / (1 + norm_x**2)
        return factor * x

    def klein_to_poincare(self, x, c):
        norm_x = torch.norm(x, dim=-1, keepdim=True)
        factor = 1 / (1 + torch.sqrt(torch.clamp(1 - norm_x**2, min=0)))#Because point feature always lie in the last dimension
        return factor * x

    #Original function only assumes x=(n,d)->(n,d+1); We make some adaptations. Needs Checking!
    def poincare_to_hyperboloid(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        sqnorm = torch.norm(x, p=2, dim=-1, keepdim=True) ** 2
        K_expand = K + sqnorm
        sqrtK_expand = 2 * sqrtK * x
        result = torch.cat([K_expand, sqrtK_expand], dim=-1) / (K - sqnorm).clamp_min(self.min_norm)
        return result

    #Original function only assumes x=(n,d)->(n,d+1); We make some adaptations. Needs Checking!
    def hyperboloid_to_poincare(self, x, c):
        K = 1. / c
        sqrtK = K ** 0.5
        d = x.size(-1) - 1
        return sqrtK * x.narrow(-1, 1, d) / (x[..., 0:1] + sqrtK)

    #I was thinking, when mapping to a different model, do we need hyperboloid_proj for constraining?
    def hyperboloid_proj(self, x, c):
        device = x.device
        dtype = x.dtype
    
        K = 1. / c
        K = torch.tensor(K.item(), device=device, dtype=dtype) 

        d = x.size(-1) - 1
        y = x.narrow(-1, 1, d)
        y_sqnorm = torch.norm(y, p=2, dim=-1, keepdim=True) ** 2

        mask = torch.ones_like(x, device=device, dtype=dtype)
        mask[..., 0] = 0

        vals = torch.zeros_like(x, device=device, dtype=dtype)
        vals[..., 0] = torch.sqrt(torch.clamp(K + y_sqnorm, min=self.eps[dtype])).squeeze(-1)

        return vals + mask * x
    
    def lorentzian_inner(self, u, v=None, *, keepdim=False):
        if v is None:
            v = u
        # Temporal Component
        time_inner = (-u[..., 0] * v[..., 0]).clamp_min(self.min_norm)
        # Spatial Component
        space_inner = torch.sum(u[..., 1:] * v[..., 1:], dim=-1, keepdim=keepdim).clamp_min(self.min_norm)
        return time_inner + space_inner

    def hyperboloid_centroid(self, x, c, w=None):
        if w is not None:
            ave = torch.einsum('bnkd,bnk->bnd', x, w)
        else:
            ave = x.mean(dim=-2)
        
        # Lorentzian norm
        lorentzian_norm = self.lorentzian_inner(ave,ave,keepdim=False).unsqueeze(-1).abs().clamp_min(1e-8).sqrt()

        # Centroid
        sqrt_neg_kappa = torch.sqrt(c)
        centroid = ave / (sqrt_neg_kappa * lorentzian_norm)

        return centroid

    def klein_midpoint(self, x, w=None):
        """
        Note:
        During inner aggregation:
        x: (n,nei_num, K, d'), the Klein features
        w: (n,nei_num,K), the Poincare/Klein kernel_feature distance
        During outer aggregation: TBC!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        x: (n,nei_num,d'), the Klein features
        w: (n,nei_num), the Poincare/Klein kernel_feature distance
        """
        def einsum_last_dim(x, y):
            """
            Operation like: 'nik,nik->ni' type einsum
            :param x: tensor x, shape of (..., k)
            :param y: tensor y, shape of (..., k)
            :return: einsum_last_dim result
            """
            x_shape = x.shape
            y_shape = y.shape
            assert x_shape == y_shape, "The shapes of x and y must match"

            num_dims = len(x_shape)

            dims = ''.join(chr(97 + i) for i in range(num_dims - 1))  # a, b, c, ...
            einsum_expr = f'{dims}k,{dims}k->{dims}'

            return torch.einsum(einsum_expr, x, y)
    
        def einsum_operation(x, y, z):
            """
            Perform einsum operation without explicitly defining dimension as nij and nijk.
            :param x: tensor of shape (..., j)
            :param y: tensor of shape (..., j)
            :param z: tensor of shape (..., j, k)
            :return: result of einsum operation
            """
            w = x * y

            return torch.einsum('...j,...jk->...k', w, z)

        klein_gamma_xi=1/(torch.sqrt(1.0-torch.norm(x,dim=-1)).abs().clamp_min(1e-8))

        if w is None:
            #Is it ok?
            w=torch.ones([x.shape[0],x.shape[1]],dtype=torch.float64)
            #This situation is mainly used for outer aggregation
        w=w.to(x.device)#From the model perspective, this x will be on cuda if specified
        lower = einsum_last_dim(klein_gamma_xi, w)
        lower = torch.where(lower > 0, lower.clamp_min(1e-8), lower.clamp_max(-1e-8))
        #Note: Giveing lower an extreme value won't affect, because the numenator is 0
        upper = einsum_operation(klein_gamma_xi,w,x)
            
        lower_expanded = lower.unsqueeze(-1).expand_as(upper)
        
        return upper / lower_expanded
        