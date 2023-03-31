#!/usr/bin/env python3

# Copyright (c) 2019-2023, Dr.-Ing. Marc Hirschvogel
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ufl

def get_eigenval_eigenvec(T, tol=1e-8):
    # analytical calulation of eigenvalues and eigenvectors for 2nd order (3x3) tensor
    # cf. Koop (2008), "Efficient numerical diagonalization of hermitian 3 x 3 matrices"

    # perturbation
    pert = 2.*tol

    # principal invariants
    I1 = ufl.tr(T)
    I2 = 0.5*(ufl.tr(T)**2. - ufl.inner(T,T))
    I3 = ufl.det(T)

    # determination of terms p and q (cf. paper)
    p = I1**2. - 3.*I2                                                           # preliminary value for p
    p = ufl.conditional(ufl.lt(p,tol), abs(p)+pert, p)                           # numerical perturbation to p, if close to zero - ensures positiveness of p
    q = (27./2.)*I3 + I1**3. - (9./2.)*I1*I2                                     # preliminary value for q
    q = ufl.conditional(ufl.lt(abs(q),tol), q+ufl.sign(q)*pert, q)               # add numerical perturbation (with sign) to value of q, if close to zero

    # determination of angle phi for calculation of roots
    phiNom2 = 27.*((1./4.)*(I2**2.)*(p-I2) + I3*((27./4.)*I3 - q))               # preliminary value for squared nominator of expression for phi
    phiNom2 = ufl.conditional(ufl.lt(phiNom2,tol), abs(phiNom2)+pert, phiNom2)   # numerical perturbation in order to guarantee non-zero nominator expression for phi
    phi = (1./3.)*ufl.atan_2(ufl.sqrt(phiNom2),q)                                # angle phi

    # polynomial roots
    lambda1 = (1./3.)*(ufl.sqrt(p)*2.*ufl.cos(phi)+I1)
    lambda2 = (1./3.)*(-ufl.sqrt(p)*(ufl.cos(phi)+ufl.sqrt(3.)*ufl.sin(phi))+I1)
    lambda3 = (1./3.)*(-ufl.sqrt(p)*(ufl.cos(phi)-ufl.sqrt(3.)*ufl.sin(phi))+I1)
    
    # eigenvectors: eq. 39 of Kopp (2008):
    v1 = ufl.cross(T[:,1] - lambda1 * ufl.as_vector([1.,0.,0.]), T[:,2] - lambda1 * ufl.as_vector([0.,1.,0.]))
    v2 = ufl.cross(T[:,1] - lambda2 * ufl.as_vector([1.,0.,0.]), T[:,2] - lambda2 * ufl.as_vector([0.,1.,0.]))
    v3 = ufl.cross(v1,v2)
    
    return [lambda1,lambda2,lambda3], [v1,v2,v3]
