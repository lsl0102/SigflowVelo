import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

class DNN(nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        self.depth = len(layers) - 1
        self.activation = nn.Tanh()

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, nn.Linear(layers[i], layers[i + 1])))
            layer_list.append(('activation_%d' % i, self.activation))

        layer_list.append(('layer_%d' % (self.depth - 1), nn.Linear(layers[-2], layers[-1])))
        self.layers = nn.Sequential(OrderedDict(layer_list))

    def forward(self, x):
        return self.layers(x)

class SingleVelocity():
    def __init__(self, adata, Outflows_expr, GEMs_expr, InGem_allscore, GemOut_regulate, iroot, layers, lr, Weight, device):

        self.adata = adata
        self.device = device
        
        # Data transfer to device
        self.Outflows_expr = Outflows_expr.clone().detach().float().to(device)
        self.GEMs_expr = GEMs_expr.clone().detach().float().to(device)
        self.InGem_allscore = InGem_allscore.clone().detach().float().to(device)
        self.regulate = GemOut_regulate.clone().detach().to(device)
        self.iroot = iroot.int().to(device)
        
        # Dimensions
        self.N_cell, self.N_TGs = self.Outflows_expr.shape
        self.N_TFs = self.GEMs_expr.shape[1]
        self.N_LRs = self.InGem_allscore.shape[2]
        self.rootcell_exp = self.Outflows_expr[self.iroot, :]

        # Latent time initialization (t1 for Resist-like, t2 for Sens-like dynamics)
        self.t1 = torch.clamp(torch.normal(mean=0.75, std=1, size=(1000,)), min=0.3, max=1).unsqueeze(1).requires_grad_(True).to(device)
        self.t2 = torch.clamp(torch.normal(mean=0.25, std=1, size=(1000,)), min=0, max=0.7).unsqueeze(1).requires_grad_(True).to(device)
        self.t = torch.cat([self.t1, self.t2], dim=0)

        # Parameters (V1, K1, V2, K2)
        self.V1 = nn.Parameter(torch.empty((self.N_TFs, self.N_LRs)).uniform_(0, 1).to(device))
        self.K1 = nn.Parameter(torch.empty((self.N_TFs, self.N_LRs)).uniform_(0, 1).to(device))
        self.V2 = nn.Parameter(torch.empty((self.N_TGs, self.N_TFs)).uniform_(0, 1).to(device))
        self.K2 = nn.Parameter(torch.empty((self.N_TGs, self.N_TFs)).uniform_(0, 1).to(device))

        self.Weight = Weight

        # DNN Setup
        self.dnn = DNN(layers).to(device)
        # Register parameters so optimizer sees them
        self.dnn.register_parameter('V1', self.V1)
        self.dnn.register_parameter('K1', self.K1)
        self.dnn.register_parameter('V2', self.V2)
        self.dnn.register_parameter('K2', self.K2)

        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=lr)

    def net_z(self):
        z0 = self.rootcell_exp.repeat(self.t.size(0), 1)
        z_and_t = torch.cat([z0, self.t], dim=1)
        z_dnn = self.dnn(z_and_t)
        
        # Calculate gradients dz/dt
        dz_dt_list = []
        for i in range(self.N_TGs):
            grad = torch.autograd.grad(
                z_dnn[:, i], self.t,
                grad_outputs=torch.ones_like(z_dnn[:, i]),
                retain_graph=True, create_graph=True
            )[0]
            dz_dt_list.append(grad)
            
        dz_dt = torch.cat(dz_dt_list, 1)
        z_dnn = torch.where(z_dnn > 0, z_dnn, torch.full_like(z_dnn, 0))
        return z_dnn, dz_dt

    def net_z_inference(self, t_input):
        """Helper for using specific time points without gradient calculation overhead for all t"""
        z0 = self.rootcell_exp.repeat(t_input.size(0), 1)
        z_and_t = torch.cat([z0, t_input], dim=1)
        z_dnn = self.dnn(z_and_t)
        z_dnn = torch.where(z_dnn > 0, z_dnn, torch.full_like(z_dnn, 0))
        return z_dnn

    def assign_latenttime(self, isSen2Res=True):
        """
        Assigns latent time based on cell groups.
        isSen2Res=True: Sensitive uses t1 (early/late logic depending on your physics), Resist uses t2
        Adjusted based on logic in original script v3/v4.
        """
        # Define time distributions based on direction
        if isSen2Res:
             # Logic from original assign_latenttime_v3
            tpoints_sens = self.t2 
            tpoints_resist = self.t1
        else:
             # Logic from original assign_latenttime_v4
            tpoints_sens = self.t1
            tpoints_resist = self.t2

        z_dnn_sens = self.net_z_inference(tpoints_sens)
        z_dnn_resist = self.net_z_inference(tpoints_resist)
        z_obs = self.Outflows_expr
        
        groups = self.adata.obs["group"]
        
        # Mappings
        mask_sens = (groups == "Sensitive").values
        mask_resist = (groups == "Resist").values
        
        fit_t = torch.zeros(z_obs.shape[0], dtype=tpoints_sens.dtype).to(self.device)
        pos = torch.zeros(z_obs.shape[0], dtype=torch.long).to(self.device)
        
        def fit_group(mask, z_dnn_ref, tpoints_ref):
            if mask.sum() > 0:
                z_obs_sub = z_obs[mask]
                # Broadcasting: (Timepoints, 1, Genes) - (1, Cells, Genes)
                loss = torch.sum((z_dnn_ref.unsqueeze(1) - z_obs_sub.unsqueeze(0)) ** 2, dim=2)
                best_pos = torch.argmin(loss, dim=0) # Index in tpoints
                best_pos = torch.clamp(best_pos, 0, tpoints_ref.size(0) - 1)
                return best_pos, tpoints_ref[best_pos].flatten()
            return None, None

        pos_sens, t_sens = fit_group(mask_sens, z_dnn_sens, tpoints_sens)
        pos_resist, t_resist = fit_group(mask_resist, z_dnn_resist, tpoints_resist)

        if t_sens is not None:
            fit_t[mask_sens] = t_sens
            pos[mask_sens] = pos_sens
        if t_resist is not None:
            fit_t[mask_resist] = t_resist
            pos[mask_resist] = pos_resist
            
        return pos, fit_t

    def calculate_y_ode(self, fit_t):
        # Calculate Initial y0
        x0 = self.InGem_allscore[self.iroot, :, :]
        Y0 = self.GEMs_expr[self.iroot, :]
        zero_y = torch.zeros(self.N_TFs, self.N_LRs).to(self.device)
        
        V1_ = torch.where(x0 > 0, self.V1, zero_y)
        K1_ = torch.where(x0 > 0, self.K1, zero_y)
        y0 = torch.sum((V1_ * x0) / (K1_ + x0 + 1e-12), dim=1) * Y0
        
        # Vectorized Hill Function Solving
        # Approximation: (((y0 + steady_state)*t)/2 + y0) * exp(-t)
        # Note: This is a simplified ODE solver based on your script
        
        # Ensure fit_t is correct shape for broadcasting
        t_col = fit_t.view(-1, 1) # (N_cells, 1)
        
        # Calculate x_i for all cells
        # Assuming InGem_allscore corresponds to cells 1-to-1
        x_all = self.InGem_allscore # (N_cells, N_TFs, N_LRs)
        Y_all = self.GEMs_expr # (N_cells, N_TFs)
        
        # Need to loop or carefully broadcast for V1/K1 masking if x_all varies per cell
        # For memory efficiency, we might keep the loop or use batch processing
        # Using loop from original script for safety, but can be optimized
        y_ode = torch.zeros((self.N_cell, self.N_TFs)).to(self.device)
        
        for i in range(self.N_cell):
            t_val = fit_t[i]
            if t_val == 0:
                y_ode[i] = y0
            else:
                x_i = self.InGem_allscore[i, :, :]
                Y_i = self.GEMs_expr[i, :]
                V1_local = torch.where(x_i > 0, self.V1, zero_y)
                K1_local = torch.where(x_i > 0, self.K1, zero_y)
                
                term_prod = torch.sum((V1_local * x_i) / (K1_local + x_i + 1e-12), dim=1) * Y_i
                term_exp = term_prod * torch.exp(t_val)
                y_ode[i] = (((y0 + term_exp) * t_val) / 2 + y0) * torch.exp(-t_val)
                
        return y_ode

    def net_f2(self, isSen2Res):
        z_dnn, dz_dt = self.net_z()
        pos, fit_t = self.assign_latenttime(isSen2Res)
        y_ode = self.calculate_y_ode(fit_t)

        zero_z = torch.zeros(self.N_TGs, self.N_TFs).to(self.device)
        V2_ = torch.where(self.regulate == 1, self.V2, zero_z)
        K2_ = torch.where(self.regulate == 1, self.K2, zero_z)
        
        # Broadcasting for (Cells, TGs, TFs)
        tmp1 = V2_.unsqueeze(0) * y_ode.unsqueeze(1)
        tmp2 = (K2_.unsqueeze(0) + y_ode.unsqueeze(1)) + 1e-12
        tmp3 = torch.sum(tmp1 / tmp2, dim=2) # Sum over TFs -> (Cells, TGs)

        # Match z_pred and dz_dt_pred to assigned time points
        z_pred_exp = z_dnn[pos]
        dz_dt_pred = dz_dt[pos]

        dz_dt_ode = tmp3 - z_pred_exp
        f = dz_dt_pred - dz_dt_ode

        return z_pred_exp, f

    def train(self, nIter=300, isSen2Res=True, verbose=True):
        self.dnn.train()
        loss_history = []
        
        for epoch in range(nIter):
            z_pred, f_pred = self.net_f2(isSen2Res)
            
            loss1 = torch.mean((self.Outflows_expr - z_pred) ** 2)
            loss2 = torch.mean(f_pred ** 2)
            
            theta = torch.cat((self.V1.flatten(), self.K1.flatten(), self.V2.flatten(), self.K2.flatten()))
            loss3 = torch.norm(theta, p=1)
            loss4 = torch.norm(theta, p=2)

            loss = (self.Weight[0] * loss1 + 
                    self.Weight[1] * loss2 + 
                    self.Weight[2] * loss3 + 
                    self.Weight[3] * loss4)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_history.append(loss.item())
            
            if verbose and (epoch % 10 == 0 or epoch == nIter-1):
                print(f'Epoch [{epoch}/{nIter}], Loss: {loss.item():.4e}')
                
        return loss_history