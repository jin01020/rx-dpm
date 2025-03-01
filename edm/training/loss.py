# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------

@persistence.persistent_class
class AuxLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, euler=False):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
       
        self.euler = euler
    def __call__(self, net, aux, images, labels=None, augment_pipe=None):
        # discrete
        # if self.k > 1: 
        #     sigma_max, sigma_min = 80, 0.002
        #     rho = 7
        #     step_indices = torch.arange(self.k, dtype=torch.float64, device=net.device)
        #     t_steps = (sigma_max ** (1 / rho) + step_indices / (self.k - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        #     try:
        #         t_steps = net.round_sigma(t_steps)
        #     except:
        #         t_steps = net.module.round_sigma(t_steps)
        #     index = (torch.ones_like(t_steps) / t_steps.shape[0]).multinomial(num_samples=images.shape[0], replacement=True)
        #     sigma = t_steps[index].reshape(-1,1,1,1).to(images.device)
        # continuous
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        sigma = torch.clamp(input=sigma, min=0.002, max=80) # hard coded

        rho = torch.randint(1, 10, (1,)).to(sigma.device)
        k = torch.FloatTensor(1).uniform_(0.08, 0.5)
        sigma_next = sample_noise_level(sigma, k=k, sigma_min=0.002, sigma_max=80, rho=rho)

        # weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        # D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        # loss = weight * ((D_yn - y) ** 2)
        with torch.no_grad():
            base = edm_step(net=net, num_steps=3, sigma=sigma, sigma_next=sigma_next, x_cur=y+n, class_labels=labels, euler=self.euler, augment_labels=augment_labels)
            rx = rx_edm_step(net=net, sigma=sigma, sigma_next=sigma_next, step_mults=[1,2], x_cur=y+n, class_labels=labels, euler=self.euler, augment_labels=augment_labels)
            real = y + n * (sigma_next / sigma)
        base_err = ((base - real) ** 2)
        rx_err = ((rx - real) ** 2)
        real.requires_grad = True
        time_cond = torch.cat((sigma.reshape(-1,1), sigma_next.reshape(-1,1)), dim=1)
        pred = aux(images, labels, time_cond).squeeze()

        decision = (rx_err - base_err < 0).float()
        loss = torch.nn.BCEWithLogistsLoss()(pred, decision)    
       
        return loss
    
#----------------------------------------------------------------------------
# sigma(t)
def sigma_ftn(t, sigma_min=0.002, sigma_max=80, rho=7):
    assert torch.all(0 <= t) and torch.all(t <= 1)
    sigma = (sigma_max**(1/rho) + t * (sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
    return torch.clamp(sigma, min=sigma_min, max=sigma_max)
# inverse of sigma(t)
def sigma_inv(sigma, sigma_min=0.002, sigma_max=80, rho=7):
    assert torch.all(sigma_min <= sigma) and torch.all(sigma <= sigma_max)
    t = (sigma**(1/rho) - sigma_max**(1/rho)) / (sigma_min**(1/rho) - sigma_max**(1/rho))
    return torch.clamp(t, min=0, max=1)

# sample projection noise level
def sample_noise_level(sigma, k=0.1, sigma_min=0.002, sigma_max=80, rho=7):
    t = sigma_inv(sigma, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
    s = torch.minimum(torch.ones_like(sigma), t + k * torch.rand_like(sigma))
    sigma_next = sigma_ftn(s)
    return torch.clamp(sigma_next, max=sigma)

def deterministic_noise_level(net, sigma, steps, sigma_min=0.002, sigma_max=80, rho=7): 
    # Time step discretization.
    step_indices = torch.arange(steps, dtype=torch.float64, device=sigma.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    try:
        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    except:
        t_steps = torch.cat([net.module.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) 
    sigma_next = [] 
    for sig in sigma.squeeze():
        sigma_next.append(t_steps[sig > t_steps].max())
    return torch.tensor(sigma_next, device=sigma.device).reshape(-1,1,1,1)

# EDM step
def edm_step(
    net, sigma, sigma_next, x_cur, 
    class_labels=None,# randn_like=torch.randn_like,
    sigma_min=0.002, sigma_max=80, rho=7, euler=False, num_steps=3, augment_labels=None,
    #S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    # try:
    #     sigma_min = max(sigma_min, net.sigma_min)
    #     sigma_max = min(sigma_max, net.sigma_max)
    # except:
    #     sigma_min = max(sigma_min, net.module.sigma_min)
    #     sigma_max = min(sigma_max, net.module.sigma_max)

    sigma_min = sigma_next
    sigma_max = sigma
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=x_cur.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
        
    # Set time step
    t_cur = sigma
    t_next = sigma_next
    try:
        t_hat = net.round_sigma(t_cur)
    except:
        t_hat = net.module.round_sigma(t_cur)
    x_hat = x_cur
    x_next = x_hat

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_hat = x_next
        # Euler step.
        denoised = net(x_hat, t_hat, class_labels, augment_labels=augment_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        
        # Apply 2nd order correction if t_next > sigma_min.
        if not euler:
            denoised = net(x_next, t_next, class_labels, augment_labels=augment_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next > sigma_min) * (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime) \
                        + (t_next <= sigma_min) * (t_next - t_hat) * d_cur

    return x_next, t_next

# EDM step
def rx_edm_step(
    net, sigma, sigma_next, x_cur, 
    class_labels=None,# randn_like=torch.randn_like,
    sigma_min=0.002, sigma_max=80, rho=7, euler=False, step_mults=[1,2], augment_labels=None,
    #S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    try:
        sigma_min = max(sigma_min, net.sigma_min)
        sigma_max = min(sigma_max, net.sigma_max)
    except:
        sigma_min = max(sigma_min, net.module.sigma_min)
        sigma_max = min(sigma_max, net.module.sigma_max)

    # Set time step
    t_cur = sigma
    t_next = sigma_next
    try:
        t_hat = net.round_sigma(t_cur)
    except:
        t_hat = net.module.round_sigma(t_cur)

    t_steps_list = []
    t_steps = torch.cat([net.round_sigma(t_hat), torch.zeros_like(t_hat)])
    t_steps_list.append(t_steps)
    t_mid = (t_cur + t_next) / 2.
    t_steps = torch.cat([net.round_sigma(t_hat), t_mid,
                         torch.zeros_like(t_steps[:1])])
    t_steps_list.append(t_steps)

    x_hat = x_cur
    recent_idx = [0] * len(step_mults)

    outs = []
    grids = []
    d1_save = 0.0
    x_next = x_hat
    x_cur_save = x_next

    for lev in range(len(step_mults)):
        
        local_steps = step_mults[lev]
        _t_steps = t_steps_list[lev] 
        x_next = x_cur_save
        t_cur = _t_steps[recent_idx[lev]]
        t_next = _t_steps[recent_idx[lev]+1]

        for j in range(local_steps):
            # Euler step.
            if lev == 0 or j > 0:
                denoised = net(x_hat, t_hat, class_labels, augment_labels=augment_labels).to(torch.float64)
                if lev == 0:
                    d1_save = denoised
            else:
                denoised = d1_save 

            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            
            # Apply 2nd order correction if t_next > sigma_min.
            if not euler:
                denoised = net(x_next, t_next, class_labels, augment_labels=augment_labels).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next > sigma_min) * (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime) \
                            + (t_next <= sigma_min) * (t_next - t_hat) * d_cur
            
            recent_idx[lev] = recent_idx[lev] + 1
            t_cur = _t_steps[recent_idx[lev]]
            t_next = _t_steps[recent_idx[lev]+1] if recent_idx[lev]+1 < len(_t_steps) else torch.zeros_like(t_cur)
        
        grids.append(_t_steps[recent_idx[lev]-local_steps:recent_idx[lev]+1])
        outs.append(x_next)

    solver = 'euler' if euler else 'heun'
    coeff = get_coeff(step_mults, solver, grids)

    x_next = 0.0
    for j in range(len(coeff)):
        x_next = x_next + coeff[j] * outs[j]

    return x_next

def get_coeff(mults, solver, grids, fix=False, adj_order=0):
    
    #assert len(grids) == len(mults)
    N = len(mults)

    if solver =='heun':
        n = 2 if adj_order < 1 else adj_order
    elif solver == 'euler':
        n = 1 if adj_order < 1 else adj_order
    elif solver == 'heun_last':
        n = 1
    # make matrix 
    A = torch.zeros(N, N).to(grids[0][0].device)
    if not fix:    
        for i in range(N):
            A[i,0] = 1.0
            for j in range(N-1):
                K = (grids[i][0] - grids[i][-1])
                if solver == 'heun_last':
                    k = (grids[i][-2] - grids[i][-1])
                    A[i,j+1] = (k/K)**(n+j+1)
                else:
                    for m in range(mults[i]):
                        k = (grids[i][m] - grids[i][m+1])
                        A[i,j+1] += (k/K)**(n+j+1)
                        #print(i, j, m, k/K)
    else:
        for i in range(N):
            A[i,0] = 1.0
            for j in range(N-1):
                K = len(grids[i]) - 1
                if solver == 'heun_last':
                    k = 1
                    A[i,j+1] = (k/K)**(n+j+1)
                else:
                    for m in range(mults[i]):
                        k = 1
                        A[i,j+1] += (k/K)**(n+j+1)
    A_inv = torch.inverse(A)
    
    #print (A_inv[0,:], A_inv[0,:].sum())
    return A_inv[0,:]          