import torch
import matplotlib.pyplot as plt
import numpy as np

def calculate_velocity(adata, model, isSen2Res=True):
    """
    Computes velocity and adds layers 'velo_Gem', 'velo_Outflow', 'velo_GemOut' to adata.
    """
    model.dnn.eval() # Set to eval mode
    device = model.device
    
    with torch.no_grad():
        # 1. Get Latent Time
        pos, fit_t = model.assign_latenttime(isSen2Res)
        adata.obs['latent_time'] = fit_t.cpu().numpy()
        
        # 2. Solve ODE for Y
        y_ode = model.calculate_y_ode(fit_t)
        
        # 3. Compute Gradients (Velocities)
        N_cell = model.N_cell
        N_TGs = model.N_TGs
        N_TFs = model.N_TFs
        
        velo_raw_Outflow = torch.zeros((N_cell, N_TGs)).to(device)
        velo_raw_Gem = torch.zeros((N_cell, N_TFs)).to(device)
        
        # Calculate Outflow Velocity
        # Formula: sum(V2 * ym / (K2 + ym)) - Outflow_expr
        for i in range(N_cell):
            y_i = y_ode[i, :]
            ym_ = model.regulate * y_i
            tmp1 = model.V2 * ym_
            tmp2 = (model.K2 + ym_) + 1e-6
            tmp3 = torch.sum(tmp1 / tmp2, dim=1) # Sum over TFs for each TG
            velo_raw_Outflow[i, :] = tmp3 - model.Outflows_expr[i, :]

        # Calculate GEM Velocity
        # Note: Original script summed dim=0 in the loop? 
        # Usually GEM velocity is degradation vs production. 
        # Assuming original script logic: sum(V2*ym/(K2+ym)) over TGs - GEM_expr?
        # Replicating original script logic exactly:
        for i in range(N_cell):
            y_i = y_ode[i, :]
            ym_ = model.regulate * y_i
            tmp1 = model.V2 * ym_
            tmp2 = (model.K2 + ym_) + 1e-6
            tmp3 = torch.sum(tmp1 / tmp2, dim=0) # Sum over TGs (rows)
            velo_raw_Gem[i, :] = tmp3 - model.GEMs_expr[i, :]
            
        # 4. Map back to full gene space
        velo_raw_Gem_full = np.zeros(adata.shape)
        velo_raw_Outflow_full = np.zeros(adata.shape)
        
        GEMs_mask = adata.var['GEMs'].astype(bool)
        GEMs_indices = np.where(GEMs_mask)[0] # Indices in adata.X
        
        Outflows_mask = adata.var['Outflows'].astype(bool)
        Outflows_indices = np.where(Outflows_mask)[0]
        
        # Assuming the model's GEMs_expr columns correspond sequentially to adata.var['GEMs'] being True
        for i, idx in enumerate(GEMs_indices):
            velo_raw_Gem_full[:, idx] = velo_raw_Gem[:, i].cpu().numpy()
            
        for i, idx in enumerate(Outflows_indices):
            velo_raw_Outflow_full[:, idx] = velo_raw_Outflow[:, i].cpu().numpy()
            
        velo_raw_GemOut = velo_raw_Gem_full + velo_raw_Outflow_full
        
        adata.layers['velo_Gem'] = velo_raw_Gem_full
        adata.layers['velo_Outflow'] = velo_raw_Outflow_full # Added separate layer
        adata.layers['velo_GemOut'] = velo_raw_GemOut
        
    return adata

def plot_loss(loss_history, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()