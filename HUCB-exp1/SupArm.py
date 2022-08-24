import json
import numpy as np

class SupArm():
    def __init__(self, suparm_id, fv, related_arms):
        self.id=suparm_id
        self.fv=fv
        self.related_arms=related_arms

class SupArmManager:
    def __init__(self, in_folder, Am,users):
        self.in_folder=in_folder
        self.am=Am
        self.suparms={}
        self.num_suparm=0
        self.users=users

    def loadArmSuparmRelation(self):
        fn=self.in_folder+'/arm_suparm_relation.txt'
        tmp_suparms={}
        with open(fn,'r') as fr:
            for line in fr:
                ele=line.strip().split('\t')
                aid=int(ele[0])
                tmp_sams=set()
                se_ele=ele[1].strip(', ').split(',')
                for se in se_ele:
                    se_a=int(se)
                    tmp_sams.add(se_a)
                wei=1.0/len(tmp_sams)

                for sa in tmp_sams:
                    try:
                        tmp=tmp_suparms[sa]
                    except:
                        tmp=tmp_suparms[sa]={}

                    try:
                        tmp=tmp_suparms[sa][aid]
                        raise AssertionError
                    except:
                        tmp_suparms[sa][aid]=wei

                    self.am.arms[aid].suparms[sa]=wei

        fv={}
        for uid, uinfo in self.users.items():
            for sup_a, alist in tmp_suparms.items():
                max_theta=float('-inf')
                for aid, wei in alist.items():
                    theta=np.dot(uinfo.theta.T,self.am.arms[aid].fv)
                    if theta>max_theta:
                        max_theta=theta
                        fv[uid]=self.am.arms[aid].fv
        # print(fv)
        for sup_a, alist in tmp_suparms.items():
           self.suparms[sup_a]=SupArm(sup_a,fv,alist)
        self.num_suparm=len(self.suparms)
