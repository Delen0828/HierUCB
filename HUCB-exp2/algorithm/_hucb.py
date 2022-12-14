import torch
import numpy as np
from ._base_algorithm import AlgorithmType, BaseAlgorithm
from ._conf import hucb_para

class HUCB(BaseAlgorithm):
    def __init__(self,users,arms,suparms, dim, device, is_update_all=False, is_update_all_attribute=False):
        super().__init__(dim, device)
        self.algorithm_type = AlgorithmType.HUCB_like
        self.is_update_all = is_update_all
        self.is_update_all_attribute = is_update_all_attribute

        # Initialize hyperparameters.
        self.lamb = hucb_para["lambda"]
        self.sigma = hucb_para["sigma"]
        self.tilde_lamb = hucb_para["tilde_lambda"]
        self.alpha = hucb_para["alpha"]
        self.tilde_alpha = hucb_para["tilde_alpha"]

        # Set placeholders.
        self.M = torch.empty((0, self.dim, self.dim), device=self.device)
        self.Minv = torch.empty((0, self.dim, self.dim), device=self.device)
        self.Y = torch.empty((0, self.dim), device=self.device)
        self.tilde_M = torch.empty((0, self.dim, self.dim), device=self.device)
        self.tilde_Minv = torch.empty((0, self.dim, self.dim), device=self.device)
        self.tilde_Y = torch.empty((0, self.dim), device=self.device)
        self.theta = torch.empty((0, self.dim), device=self.device)
        self.tilde_theta = torch.empty((0, self.dim), device=self.device)
        self.count = {}
        self.supcount = {}
        for uid in range(users):
            for sid, suparm in suparms.suparms.items():
                self.supcount[(uid, sid)] = 1  
            for aid, arm in arms.arms.items():
                self.count[(uid, aid)] = 1
        self.need_switch = False	
        
    def add_user(self, user_idx):
        # Initialize item parameters.
        new_M = (1 - self.lamb) * torch.eye(self.dim, device=self.device)
        new_Minv = torch.eye(self.dim, device=self.device) / (1 - self.lamb)
        new_Y = torch.zeros(self.dim, device=self.device)

        # Initialize attribute parameters.
        new_tilde_M = self.tilde_lamb * torch.eye(self.dim, device=self.device)
        new_tilde_Minv = torch.eye(self.dim, device=self.device) / self.tilde_lamb
        new_tilde_Y = torch.zeros(self.dim, device=self.device)

        # Initialize user parameters.
        new_theta = torch.zeros(self.dim, device=self.device)
        new_tilde_theta = torch.zeros(self.dim, device=self.device)

        # Record the incoming user.
        self.M = torch.cat((self.M, new_M.unsqueeze(0)))
        self.Minv = torch.cat((self.Minv, new_Minv.unsqueeze(0)))
        self.Y = torch.cat((self.Y, new_Y.unsqueeze(0)))
        self.tilde_M = torch.cat((self.tilde_M, new_tilde_M.unsqueeze(0)))
        self.tilde_Minv = torch.cat((self.tilde_Minv, new_tilde_Minv.unsqueeze(0)))
        self.tilde_Y = torch.cat((self.tilde_Y, new_tilde_Y.unsqueeze(0)))
        self.theta = torch.cat((self.theta, new_theta.unsqueeze(0)))
        self.tilde_theta = torch.cat((self.tilde_theta, new_tilde_theta.unsqueeze(0)))

        self.user_idx_map[user_idx] = self.num_users
        self.num_users += 1

    def decide(self, user_idx, X, selected_arm_idxs):
        if user_idx not in self.user_idx_map:
            self.add_user(user_idx)
        user_idx = self.user_idx_map[user_idx]
        arm_id= selected_arm_idxs[torch.argmax(self._get_prob(user_idx, X[selected_arm_idxs])).item()]
        self.count[(user_idx,arm_id)]+=1
        return arm_id

    def decide_attribute(self, user_idx, X,tilde_X, armpool,arm_to_suparm):
        if user_idx not in self.user_idx_map:
            self.add_user(user_idx)
        user_idx = self.user_idx_map[user_idx]

        bestarm=armpool[torch.argmax(self._get_prob(user_idx, X[armpool]))]
        if len(arm_to_suparm[bestarm])!=0:
            sup=arm_to_suparm[bestarm][0]
            # print('sup',sup)
        else:
            sup=torch.argmax(self._get_credit(user_idx, X, tilde_X)).item()
        
        self.supcount[(user_idx,sup)]+=1
        self.count[(user_idx,bestarm)]+=1
        return sup,bestarm

    def update(self, user_idx, x, y):
        user_idx = self.user_idx_map[user_idx]

        if self.is_update_all:
            self._update_all(x, y)
        else:
            self._update(user_idx, x, y)

    def update_attribute(self, user_idx, tilde_x, tilde_y):
        user_idx = self.user_idx_map[user_idx]

        if self.is_update_all_attribute:
            self._update_attribute_all(tilde_x, tilde_y)
        else:
            self._update_attribute(user_idx, tilde_x, tilde_y)

    def _get_prob(self, user_idx, X):
        # Please refer to Eq.7.
        theta = self.theta[user_idx]
        Minv, tilde_Minv = self.Minv[user_idx], self.tilde_Minv[user_idx]

        X_Minv = torch.mm(X, Minv)
        var1 = (X_Minv * X).sum(dim=1).sqrt()
        var2 = (torch.mm(X_Minv, tilde_Minv) * X_Minv).sum(dim=1).sqrt()
        mean = torch.mv(X, theta)

        return mean + self.lamb * self.alpha * var1 + (1 - self.lamb) * self.tilde_alpha * var2

    def _get_uncertainty(self, user_idx, X):
        # Please refer to Eq.7.
        Minv, tilde_Minv = self.Minv[user_idx], self.tilde_Minv[user_idx]

        X_Minv = torch.mm(X, Minv)
        var1 = (X_Minv * X).sum(dim=1).sqrt()
        var2 = (torch.mm(X_Minv, tilde_Minv) * X_Minv).sum(dim=1).sqrt()

        return self.lamb * self.alpha * var1 + (1 - self.lamb) * self.tilde_alpha * var2

    def _get_credit(self, user_idx, X, tilde_X):
        # Please refer to Eq.8.
        Minv, tilde_Minv = self.Minv[user_idx], self.tilde_Minv[user_idx]

        tilde_Minv_tilde_X = torch.mm(tilde_Minv, tilde_X.T)
        result_a = torch.chain_matmul(X, Minv, tilde_Minv_tilde_X)
        result_b = 1 + (tilde_X.T * tilde_Minv_tilde_X).sum(dim=0)
        norm_M = result_a.norm(dim=0)

        return norm_M * norm_M / result_b

    def _update(self, user_idx, x, y):
        self.M[user_idx] += torch.ger(x, x)
        self.Minv[user_idx], _ = self._update_Minv(self.Minv[user_idx], x)
        self.Y[user_idx] += y * x
        self.theta[user_idx] = torch.mv(self.Minv[user_idx], self.Y[user_idx])

    def _update_all(self, x, y):
        self.M += (self.lamb * torch.ger(x, x)).unsqueeze(0).repeat(self.num_users, 1, 1)
        self.Minv, _ = self._update_Minv_all(self.Minv, (self.lamb ** 0.5) * x)
        self.Y += (self.lamb * y * x).unsqueeze(0).repeat(self.num_users, 1)
        self.theta = torch.bmm(self.Minv, (self.Y + (1 - self.lamb) * self.tilde_theta).unsqueeze(2)).squeeze(2)

    def _update_attribute(self, user_idx, tilde_x, tilde_y):
        self.tilde_M[user_idx] += torch.ger(tilde_x, tilde_x)
        self.tilde_Minv[user_idx], _ = self._update_Minv(self.tilde_Minv[user_idx], tilde_x)
        self.tilde_Y[user_idx] += tilde_y * tilde_x
        self.tilde_theta[user_idx] = torch.mv(self.tilde_Minv[user_idx], self.tilde_Y[user_idx])
        self.theta[user_idx] = torch.mv(self.Minv[user_idx], self.Y[user_idx] + (1 - self.lamb) * self.tilde_theta[user_idx])

    def _update_attribute_all(self, tilde_x, tilde_y):
        self.tilde_M += (torch.ger(tilde_x, tilde_x)).unsqueeze(0).repeat(self.num_users, 1, 1)
        self.tilde_Minv, _ = self._update_Minv_all(self.tilde_Minv, tilde_x)
        self.tilde_Y += (tilde_y * tilde_x).unsqueeze(0).repeat(self.num_users, 1)
        self.tilde_theta = torch.bmm(self.tilde_Minv, self.tilde_Y.unsqueeze(2)).squeeze(2)
        self.theta = torch.bmm(self.Minv, self.Y.unsqueeze(2)).squeeze(2) + (1 - self.lamb) * self.tilde_theta
    
    def update_switch(self, uid, it, suparm, arm, X, k):
        rho_sup = np.sqrt(np.log(it+1)/(self.supcount[(uid, suparm)]))  # ??
        rho_arm = np.sqrt(np.log(it+1)/(self.count[(uid, arm)]))
        a_pta = float(torch.max(self._get_prob(uid, X)))
        s_pta = k*a_pta
        print('count',self.supcount[(uid, suparm)],self.count[(uid, arm)])
        print('rho',rho_sup,rho_arm)
        print('Differ:', s_pta+rho_sup*k, a_pta-rho_arm*k, it)
        if s_pta+rho_sup*k < a_pta-rho_arm*k and it > 50:
            self.need_switch = True
            print("in: need switch")
        else:
            self.need_switch = False

    def need_switch(self):
        return self.need_switch		