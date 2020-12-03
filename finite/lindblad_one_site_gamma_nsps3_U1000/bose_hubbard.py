#!/usr/bin/env python

import numpy as np
import scipy.linalg

def make_op(nsps):
    op_id = np.eye(nsps,dtype=np.float64)
    op_a = np.zeros((nsps,nsps),dtype=np.float64)
    op_adag = np.zeros((nsps,nsps),dtype=np.float64)
    op_n = np.zeros((nsps,nsps),dtype=np.float64)
    op_n2 = np.zeros((nsps,nsps),dtype=np.float64)
    for i in range(nsps-1):
        op_a[i,i+1] = np.sqrt(i+1)
        op_adag[i+1,i] = np.sqrt(i+1)
    for i in range(nsps):
        op_n[i,i] = i
        op_n2[i,i] = i**2
    return op_id, op_a, op_adag, op_n, op_n2

def make_list_parameters(J,U,mu,phi,L):
    list_J = np.array([J for i in range(L)])
    list_U = np.array([U for i in range(L)])
    list_mu = np.array([mu for i in range(L)])
    list_phi = np.array([phi for i in range(L)])
    return list_J, list_U, list_mu, list_phi

def make_list_ham(op_id,op_a,op_adag,op_n,op_n2,list_J,list_U,list_mu,list_phi):
    return [- J*((phip+phim).conj()*op_a + (phip+phim)*op_adag)
        + 0.5*U*op_n2 - (0.5*U+mu)*op_n \
        for J,U,mu,phip,phim \
        in zip(list_J,list_U,list_mu,np.roll(list_phi,-1),np.roll(list_phi,1))]

def _eigh_GS(H):
    ene, vec = scipy.linalg.eigh(H)
    return ene[0], vec[:,0]

def calc_list_ene_vec(list_ham,L):
    enevec = np.array([_eigh_GS(list_ham[i]) for i in range(L)])
    return enevec[:,0], enevec[:,1]

def calc_list_phi(op_a,list_vec,L):
    return np.array([list_vec[i].dot(op_a.dot(list_vec[i]))/np.linalg.norm(list_vec[i])**2 for i in range(L)])

def calc_list_phys(op_a,op_n,op_n2,list_vec,L):
    list_norm2 = np.array([np.linalg.norm(list_vec[i])**2 for i in range(L)])
    list_phi = np.array([list_vec[i].dot(op_a.dot(list_vec[i]))/list_norm2[i] for i in range(L)])
    list_n = np.array([list_vec[i].dot(op_n.dot(list_vec[i]))/list_norm2[i] for i in range(L)])
    list_n2 = np.array([list_vec[i].dot(op_n2.dot(list_vec[i]))/list_norm2[i] for i in range(L)])
    return list_phi, list_n, list_n2

def calc_list_gs(op_id,op_a,op_adag,op_n,op_n2,list_J,list_U,list_mu,list_phi,L):
    Nstep = 1000
    list_phi_old = list_phi
    list_phi_new = list_phi
    phi_eps = 1e-12
    for step in range(Nstep):
        list_ham = make_list_ham(op_id,op_a,op_adag,op_n,op_n2,list_J,list_U,list_mu,list_phi_old)
        list_ene, list_vec = calc_list_ene_vec(list_ham,L)
        list_phi_new = calc_list_phi(op_a,list_vec,L)
        list_dphi = np.abs(list_phi_new - list_phi_old)
        list_phi_old = list_phi_new
#        print(step,list_phi_new[0],list_dphi[0])
        if np.sum(list_dphi)/L < phi_eps:
            break
    list_ham = make_list_ham(op_id,op_a,op_adag,op_n,op_n2,list_J,list_U,list_mu,list_phi_new)
    list_ene, list_vec = calc_list_ene_vec(list_ham,L)
    list_phi, list_n, list_n2 = calc_list_phys(op_a,op_n,op_n2,list_vec,L)
    return list_ene, list_vec, list_phi, list_dphi, list_n, list_n2

def list_vec2list_rho(list_vec,L):
    return np.array([np.array([list_vec[i]]).conj().T.dot(np.array([list_vec[i]])) for i in range(L)])

def calc_list_drhodt(list_gamma,op_n,op_n2,list_ham,list_rho,L):
    return np.array([\
        - 1j * (list_ham[i].dot(list_rho[i]) - list_rho[i].dot(list_ham[i])) \
        - 0.5 * list_gamma[i] * (op_n2.dot(list_rho[i]) + list_rho[i].dot(op_n2) \
        - 2.0 * op_n.dot(list_rho[i].dot(op_n)))\
        for i in range(L)])

# fourth order Runge-Kutta
def calc_RK(dt,list_gamma,op_n,op_n2,list_ham,list_rho,L):
    k1 = dt * calc_list_drhodt(list_gamma,op_n,op_n2,list_ham,list_rho,L)
    k2 = dt * calc_list_drhodt(list_gamma,op_n,op_n2,list_ham,list_rho+0.5*k1,L)
    k3 = dt * calc_list_drhodt(list_gamma,op_n,op_n2,list_ham,list_rho+0.5*k2,L)
    k4 = dt * calc_list_drhodt(list_gamma,op_n,op_n2,list_ham,list_rho+k3,L)
    return list_rho + (k1+2*k2+2*k3+k4)/6.0

## prepare rho = |+><+| = [[.5,.5],[.5,.5]]

def main():
    L = 10
    J = 1.0
    U = 1000.0
## nsps = n_{max states per site} = n_{max occupation} + 1
#    nsps = 11
    nsps = 3
    mu = 0.371
    phi = nsps
#    gamma = 0.0
    gamma = 0.1
#
    op_id, op_a, op_adag, op_n, op_n2 = make_op(nsps)
#
    list_J, list_U, list_mu, list_phi = make_list_parameters(J,U,mu,phi,L)
#    list_ene, list_vec, list_phi, list_dphi, list_n, list_n2 = \
#        calc_list_gs(op_id,op_a,op_adag,op_n,op_n2,list_J,list_U,list_mu,list_phi,L)
##    print(list_ene,list_vec,list_phi,list_dphi,list_n,list_n2)
##    print(list_ene[0],list_vec[0],list_phi[0],list_dphi[0],list_n[0],list_n2[0])
##    print(list_ene[L-1],list_vec[L-1],list_phi[L-1],list_dphi[L-1],list_n[L-1],list_n2[L-1])
#    print("#",list_ene[0],list_phi[0],list_dphi[0],list_n[0],list_n2[0])
#    print()
#
##    Js = np.linspace(0,0.1,101)
#    Js = np.linspace(0,0.2,101)
#    for J in Js:
#        list_J, list_U, list_mu, list_phi = make_list_parameters(J,U,mu,phi,L)
#        list_ene, list_vec, list_phi, list_dphi, list_n, list_n2 = \
#            calc_list_gs(op_id,op_a,op_adag,op_n,op_n2,list_J,list_U,list_mu,list_phi,L)
#        print(J,list_ene[0],list_phi[0],list_dphi[0],list_n[0],list_n2[0])
#
    vec = np.array([1.0,1.0,0.0])/np.sqrt(2.0)
    list_vec = np.array([vec for i in range(L)])
    list_rho = list_vec2list_rho(list_vec,L)
#    print(list_rho[0])
#    print([np.trace(list_rho[i]) for i in range(L)])
    list_phi = np.array([np.trace(list_rho[i].dot(op_a)) for i in range(L)])
##    print([np.trace(list_rho[i].dot(op_a)) for i in range(L)])
#    print(list_phi)
    list_ham = make_list_ham(op_id,op_a,op_adag,op_n,op_n2,list_J,list_U,list_mu,list_phi)
#    print([np.trace(list_rho[i].dot(list_ham[i])) for i in range(L)])
#    print()
#
#    list_gamma = np.array([gamma for i in range(L)])
#    list_ham = make_list_ham(op_id,op_a,op_adag,op_n,op_n2,list_J,list_U,list_mu,list_phi)
#    list_drhodt = calc_list_drhodt(list_gamma,op_n,op_n2,list_ham,list_rho,L)
#    print(list_drhodt[0])
#    print([np.trace(list_drhodt[i]) for i in range(L)])
#    print([np.trace(list_drhodt[i].dot(op_a)) for i in range(L)])
#    list_ham = make_list_ham(op_id,op_a,op_adag,op_n,op_n2,list_J,list_U,list_mu,list_phi)
#    print([np.trace(list_drhodt[i].dot(list_ham[i])) for i in range(L)])
#    print()
#
    mu0 = 0.0
    list_mu0 = np.array([mu0 for i in range(L)])
#    dt = 0.01
#    dt = 0.005
    dt = 0.001
#    Nsteps = 100000
#    Nsteps = 10000 ## 15sec
#    Nsteps = 100000
    Nsteps = 1000000
#    list_gamma = np.array([gamma for i in range(L)])
    list_gamma = np.zeros(L)
#    list_gamma[L//2] = gamma
    list_gamma[0] = gamma
#    list_ham = make_list_ham(op_id,op_a,op_adag,op_n,op_n2,list_J,list_U,list_mu,list_phi)
    list_ham = make_list_ham(op_id,op_a,op_adag,op_n,op_n2,list_J,list_U,list_mu0,list_phi) ## no chemical potential
    list_rho = list_vec2list_rho(list_vec,L)
#    for steps in range(Nsteps):
    for steps in range(Nsteps+1):
        list_rho1 = calc_RK(dt,list_gamma,op_n,op_n2,list_ham,list_rho,L)
        list_rho1_proj_site_diag = np.array([np.abs(list_rho1[0][i,i]) for i in range(nsps)])
        list_phi1 = np.array([np.trace(list_rho1[i].dot(op_a)) for i in range(L)])
        list_norm = np.array([np.trace(list_rho1[i]) for i in range(L)])
#        list_ham = make_list_ham(op_id,op_a,op_adag,op_n,op_n2,list_J,list_U,list_mu,list_phi)
        list_ham = make_list_ham(op_id,op_a,op_adag,op_n,op_n2,list_J,list_U,list_mu0,list_phi1)
        list_ene = np.array([np.trace(list_rho1[i].dot(list_ham[i])) for i in range(L)])
        list_n = np.array([np.trace(list_rho1[i].dot(op_n)) for i in range(L)])
        list_rho = list_rho1
        list_phi = list_phi1
        if steps%100 == 0:
#            print(dt*steps,np.abs(np.trace(list_rho1[0])),np.abs(list_phi1[0]),np.abs(list_ene[0]))
            print("## rho1_proj_site_diag ",dt*steps," ".join(str(x) for x in np.abs(list_rho1_proj_site_diag)))
            print("## norm ",dt*steps," ".join(str(x) for x in np.abs(list_norm)))
            print("## phi ",dt*steps," ".join(str(x) for x in np.abs(list_phi1)))
            print("## n ",dt*steps," ".join(str(x) for x in np.abs(list_n)))
            print("## ene ",dt*steps," ".join(str(x) for x in np.abs(list_ene)))

if __name__ == "__main__":
    main()
