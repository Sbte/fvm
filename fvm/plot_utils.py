import numpy
import matplotlib.pyplot as plt

from fvm import utils
from fvm.utils import create_state_mtx # noqa: F401

def get_meshgrid(interface, x=None, y=None):
    if x is None:
        x = interface.discretization.x[:-3]
    if y is None:
        if interface.discretization.ny > 1:
            y = interface.discretization.y[:-3]
        else:
            y = interface.discretization.z[:-3]

    return numpy.meshgrid(x, y)

def plot_velocity_magnitude(state, interface, axis=2):
    m = utils.compute_velocity_magnitude(state, interface, axis)

    x, y = get_meshgrid(interface)

    fig1, ax1 = plt.subplots()
    cs = ax1.contourf(x, y, m.transpose(), 15)
    fig1.colorbar(cs)

    ax1.vlines(x[0, :], *y[[0, -1], 0], colors='0.3', linewidths=0.5)
    ax1.hlines(y[:, 0], *x[0, [0, -1]], colors='0.3', linewidths=0.5)

    plt.show()

def plot_streamfunction(state, interface):
    psi = utils.compute_streamfunction(state, interface)

    x, y = get_meshgrid(interface)

    fig1, ax1 = plt.subplots()
    cs = ax1.contourf(x, y, psi.transpose(), 15)
    fig1.colorbar(cs)

    ax1.vlines(x[0, :], *y[[0, -1], 0], colors='0.3', linewidths=0.5)
    ax1.hlines(y[:, 0], *x[0, [0, -1]], colors='0.3', linewidths=0.5)

    plt.show()

def plot_value(t, interface=None, x=None, y=None):
    x, y = get_meshgrid(interface, x, y)

    fig1, ax1 = plt.subplots()
    cs = ax1.contourf(x, y, t.transpose(), 15)
    fig1.colorbar(cs)

    ax1.vlines(x[0, :], *y[[0, -1], 0], colors='0.3', linewidths=0.5)
    ax1.hlines(y[:, 0], *x[0, [0, -1]], colors='0.3', linewidths=0.5)

    plt.show()
