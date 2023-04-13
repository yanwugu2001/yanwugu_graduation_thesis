import numpy as np
import pandas as pd
import numba as nb

def l_bin(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    '''
    x: (b, n)
    y: (b, )
    w: (1, n)
    '''
    return l_bin_numba(x, y, w)

@nb.njit
def l_bin_numba(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    b, n = x.shape
    l = 0
    for i in range(b):
        z = 0
        for j in range(n):
            z += x[i, j] * w[i]
        l += np.log(1+np.exp(-y[i]*z))
    return l

def l_bin_grad(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    return l_bin_grad_numba(x, y, w)

@nb.njit
def l_bin_grad_numba(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    b, n = x.shape
    g = np.zeros((1, n))
    for i in range(b):
        z = 0
        for j in range(n):
            z += x[i, j] * w[i]
        g += -y[i] * np.exp(-y[i]*z) / (1+np.exp(-y[i]*z)) * x[i, :]
    return g

def l_bin_hess(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    return l_bin_hess_numba(x, y, w)

@nb.njit
def l_bin_hess_numba(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    b, n = x.shape
    h = np.zeros((n, n))
    for i in range(b):
        z = 0
        for j in range(n):
            z += x[i, j] * w[i]
        h += np.exp(-y[i]*z) / (1+np.exp(-y[i]*z))**2 * np.outer(x[i, :], x[i, :])
    return h

def l_multi(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    '''
    x: (b, n)
    y: (b, )
    w: (1, n*k)
    '''
    return l_multi_numba(x, y, w)

@nb.njit
def l_multi_numba(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    b, n = x.shape
    k = w.shape[0] // n
    l = 0
    for i in range(b):
        z = np.zeros(k)
        for _k in range(k):
            for j in range(n):
                z[_k] += x[i, j] * w[j+ n*_k]
        ez = 0
        for _k in range(k):
            ez += np.exp(z[_k])
        l += np.log(ez) - z[y[i]]
    return l

def l_multi_grad(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    return l_multi_grad_numba(x, y, w)

@nb.njit
def l_multi_grad_numba(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    b, n = x.shape
    k = w.shape[0] // n
    g = np.zeros((1, n*k))
    for i in range(b):
        z = np.zeros(k)
        for _k in range(k):
            for j in range(n):
                z[_k] += x[i, j] * w[j+ n*_k]
        ez = 0
        for _k in range(k):
            ez += np.exp(z[_k])
        for _k in range(k):
            for j in range(n):
                g[0, j + n*_k] += np.exp(z[_k]) / ez * x[i, j]
        for j in range(n):
            g[0, j + n*y[i]] -= x[i, j]
    return g

def l_multi_hess(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    return l_multi_hess_numba(x, y, w)

@nb.njit
def l_multi_hess_numba(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    b, n = x.shape
    k = w.shape[0] // n
    h = np.zeros((n*k, n*k))
    for i in range(b):
        z = np.zeros(k)
        for _k in range(k):
            for j in range(n):
                z[_k] += x[i, j] * w[j+ n*_k]   
        ez = 0
        for _k in range(k):
            ez += np.exp(z[_k])
        for _k in range(k):
            for _l in range(k):
                for j in range(n):
                    for _j in range(n):
                        h[j + n*_k, _j + n*_l] += np.exp(z[_k]) / ez * np.exp(z[_l]) / ez * x[i, j] * x[i, _j]
        for _k in range(k):
            for j in range(n):
                for _j in range(n):
                    h[j + n*_k, _j + n*y[i]] -= np.exp(z[_k]) / ez * x[i, j] * x[i, _j]
    return h

def Gradient_Descent(data_x, data_y, learning_rate, type='bin'):
    if type=='bin':
        w = np.zeros((1, data_x.shape[1]))
        ws = [w]
        gs = []
        ls = []
        for i in range(data_x.shape[0]):
            w = ws[-1]
            ls.append(l_bin(data_x[i, :].reshape(1, -1), data_y[i], w))
            g = l_bin_grad(data_x[i, :].reshape(1, -1), data_y[i], w)
            w = w - learning_rate * g
            w = w / np.linalg.norm(w)
            ws.append(w)
            gs.append(g)
        return ws, gs, ls
    elif type=='multi':
        w = np.zeros((1, data_x.shape[1]))
        ws = [w]
        gs = []
        ls = []
        for i in range(data_x.shape[0]):
            w = ws[-1]
            ls.append(l_multi(data_x[i, :].reshape(1, -1), data_y[i], w))
            g = l_multi_grad(data_x[i, :].reshape(1, -1), data_y[i], w)
            w = w - learning_rate * g
            w = w / np.linalg.norm(w)
            ws.append(w)
            gs.append(g)
        return ws, gs, ls
            

def Mini_Batch_Gradient_Descent(data_x, data_y, learning_rate, batch_size, type='bin'):
    if type=='bin':
        w = np.zeros((1, data_x.shape[1]))
        ws = [w]
        gs = []
        ls = []
        for i in range(data_x.shape[0] // batch_size):
            w = ws[-1]
            ls.append(l_bin(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w))
            g = l_bin_grad(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w)
            w = w - learning_rate * g
            w = w / np.linalg.norm(w)
            ws.append(w)
            gs.append(g)
        return ws, gs, ls
    elif type=='multi':
        w = np.zeros((1, data_x.shape[1]))
        ws = [w]
        gs = []
        ls = []
        for i in range(data_x.shape[0] // batch_size):
            w = ws[-1]
            ls.append(l_multi(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w))
            g = l_multi_grad(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w)
            w = w - learning_rate * g
            w = w / np.linalg.norm(w)
            ws.append(w)
            gs.append(g)
        return ws, gs, ls
    
def Gradient_Descent_with_Momentum(data_x, data_y, learning_rate, batch_size, momentum_factor, type='bin'):
    if type=='bin':
        w = np.zeros((1, data_x.shape[1]))
        ws = [w]
        gs = []
        ls = []
        v = np.zeros((1, data_x.shape[1]))
        for i in range(data_x.shape[0] // batch_size):
            w = ws[-1]
            ls.append(l_bin(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w))
            g = l_bin_grad(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w)
            v = momentum_factor * v + (1 - momentum_factor) * g
            w = w - learning_rate * v
            w = w / np.linalg.norm(w)
            ws.append(w)
            gs.append(g)
        return ws, gs, ls
    elif type=='multi':
        w = np.zeros((1, data_x.shape[1]))
        ws = [w]
        gs = []
        ls = []
        v = np.zeros((1, data_x.shape[1]))
        for i in range(data_x.shape[0] // batch_size):
            w = ws[-1]
            ls.append(l_multi(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w))
            g = l_multi_grad(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w)
            v = momentum_factor * v + (1 - momentum_factor) * g
            w = w - learning_rate * v
            w = w / np.linalg.norm(w)
            ws.append(w)
            gs.append(g)
        return ws, gs, ls
    
def Nestrov_Accelerated_Gradient(data_x, data_y, learning_rate, batch_size, momentum_factor, type='bin'):
    if type=='bin':
        w = np.zeros((1, data_x.shape[1]))
        ws = [w]
        gs = []
        ls = []
        v = np.zeros((1, data_x.shape[1]))
        for i in range(data_x.shape[0] // batch_size):
            w = ws[-1]
            ls.append(l_bin(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w))
            g = l_bin_grad(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w - momentum_factor * v)
            v = momentum_factor * v + (1 - momentum_factor) * g
            w = w - learning_rate * v
            w = w / np.linalg.norm(w)
            ws.append(w)
            gs.append(g)
        return ws, gs, ls
    elif type=='multi':
        w = np.zeros((1, data_x.shape[1]))
        ws = [w]
        gs = []
        ls = []
        v = np.zeros((1, data_x.shape[1]))
        for i in range(data_x.shape[0] // batch_size):
            w = ws[-1]
            ls.append(l_multi(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w))
            g = l_multi_grad(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w - momentum_factor * v)
            v = momentum_factor * v + (1 - momentum_factor) * g
            w = w - learning_rate * v
            w = w / np.linalg.norm(w)
            ws.append(w)
            gs.append(g)
        return ws, gs, ls
    
def AdaGrad(data_x, data_y, learning_rate, batch_size, momentum_factor, type='bin'):
    if type=='bin':
        w = np.zeros((1, data_x.shape[1]))
        ws = [w]
        gs = []
        ls = []
        for i in range(data_x.shape[0] // batch_size):
            r = 0 
            w = ws[-1]
            ls.append(l_bin(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w))
            g = l_bin_grad(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w)
            r = r + np.linalg.norm(g) ** 2
            w = w - learning_rate / (np.sqrt(r) + 1e-8) * g
            w = w / np.linalg.norm(w)
            ws.append(w)
            gs.append(g)
        return ws, gs, ls
    elif type=='multi':
        w = np.zeros((1, data_x.shape[1]))
        ws = [w]
        gs = []
        ls = []
        for i in range(data_x.shape[0] // batch_size):
            r = 0 
            w = ws[-1]
            ls.append(l_multi(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w))
            g = l_multi_grad(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w)
            r = r + np.linalg.norm(g) ** 2
            w = w - learning_rate / (np.sqrt(r) + 1e-8) * g
            w = w / np.linalg.norm(w)
            ws.append(w)
            gs.append(g)
        return ws, gs, ls
    
def Adam(data_x, data_y, learning_rate, batch_size, beta1, beta2, type='bin'):
    if type == 'bin':
        w = np.zeros((1, data_x.shape[1]))
        ws = [w]
        gs = []
        ls = []
        m = np.zeros((1, data_x.shape[1]))
        v = 0
        for i in range(data_x.shape[0] // batch_size):
            w = ws[-1]
            ls.append(l_bin(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w))
            g = l_bin_grad(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * np.linalg.norm(g) ** 2
            w = w - learning_rate / (np.sqrt(v / 1-beta2 ** i) + 1e-8) * m / (1-beta1 ** i)
            w = w / np.linalg.norm(w)
            ws.append(w)
            gs.append(g)
        return ws, gs, ls
    elif type == 'multi':
        w = np.zeros((1, data_x.shape[1]))
        ws = [w]
        gs = []
        ls = []
        m = np.zeros((1, data_x.shape[1]))
        v = 0
        for i in range(data_x.shape[0] // batch_size):
            w = ws[-1]
            ls.append(l_multi(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w))
            g = l_multi_grad(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * np.linalg.norm(g) ** 2
            w = w - learning_rate / (np.sqrt(v / 1-beta2 ** i) + 1e-8) * m / (1-beta1 ** i)
            w = w / np.linalg.norm(w)
            ws.append(w)
            gs.append(g)
        return ws, gs, ls
    
def Newton_Method(data_x, data_y, gamma, epsilon, batch_size, type='bin'):
    if type == 'bin':
        w = np.zeros((1, data_x.shape[1]))
        ws = [w]
        gs = []
        ls = []
        H = epsilon * np.eye(data_x.shape[1])
        for i in range(data_x.shape[0] // batch_size):
            w = ws[-1]
            ls.append(l_bin(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w))
            g = l_bin_grad(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w)
            H = H + np.outer(g, g)
            w = w - 1 / gamma * np.linalg.inv(H) @ g
            w = w / np.linalg.norm(w)
            ws.append(w)
            gs.append(g)
        return ws, gs, ls
    elif type == 'multi':
        w = np.zeros((1, data_x.shape[1]))
        ws = [w]
        gs = []
        ls = []
        H = epsilon * np.eye(data_x.shape[1])
        for i in range(data_x.shape[0] // batch_size):
            w = ws[-1]
            ls.append(l_multi(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w))
            g = l_multi_grad(data_x[i*batch_size:(i+1)*batch_size, :], data_y[i*batch_size:(i+1)*batch_size], w)
            H = H + np.outer(g, g)
            w = w - 1 / gamma * np.linalg.inv(H) @ g
            w = w / np.linalg.norm(w)
            ws.append(w)
            gs.append(g)
        return ws, gs, ls
