#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 09:38:54 2020

@author: eke001
"""

import numpy as np

def compare_arrays(a, b, tol=1e-4, sort=True):
    """ Compare two arrays and check that they are equal up to a column permutation.

    Typical usage is to compare coordinate arrays.

    Parameters:
        a, b (np.array): Arrays to be compared. W
        tol (double, optional): Tolerance used in comparison.
        sort (boolean, defaults to True): Sort arrays columnwise before comparing

    Returns:
        True if there is a permutation ind so that all(a[:, ind] == b).
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)

    if not np.all(a.shape == b.shape):
        return False

    if sort:
        a = np.sort(a, axis=0)
        b = np.sort(b, axis=0)

    for i in range(a.shape[1]):
        dist = np.sum((b - a[:, i].reshape((-1, 1))) ** 2, axis=0)
        if dist.min() > tol:
            return False
    for i in range(b.shape[1]):
        dist = np.sum((a - b[:, i].reshape((-1, 1))) ** 2, axis=0)
        if dist.min() > tol:
            return False
    return True

def compare_arrays_varying_size(a, b, tol=1e-4, sort=True):
    
    if not np.all(a.shape == b.shape):
        return False
    
    if sort:
        for i in range(a.shape[-1]):
            a[i] = np.sort(a[i])
        for i in range(b.shape[-1]):
            b[i] = np.sort(b[i])
            
    for ia in range(a.shape[0]):
        a_found = False
        for ib in range(b.shape[0]):
            if a[ia].size != b[ib].size:
                continue
            dist = np.sum((a[ia] - b[ib])**2)
            if dist < tol**2:
                a_found = True
                
        if not a_found:
            return False
        
    return True
    