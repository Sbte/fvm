import numpy
import pickle

import matplotlib.pyplot as plt

from transiflow.interface.SciPy import Interface

from transiflow import Continuation
from transiflow import plot_utils
from transiflow import utils


class Data:
    def __init__(self):
        self.mu = []
        self.value = []

    def append(self, mu, value):
        self.mu.append(mu)
        self.value.append(value)

    def callback(self, interface, x, mu):
        self.append(mu, numpy.max(utils.compute_streamfunction(x, interface)))


def generate_plots(interface, x, name, beta):
    '''Generate plots for the stream function, vorticity and salinity and write them to a file'''
    psi_max = numpy.max(utils.compute_streamfunction(x, interface))

    plot_utils.plot_streamfunction(
        x, interface, title=f'Stream function at $\\beta={beta:.2f}$ and $\\Psi_\\max={psi_max:.2e}$',
        legend=False, grid=False, show=False)
    plt.savefig(f'streamfunction_{name}_{beta:.2f}_{psi_max:.2e}.eps')
    plt.close()

    plot_utils.plot_vorticity(
        x, interface, title=f'Vorticity at $\\beta={beta:.2f}$ and $\\Psi_\\max={psi_max:.2e}$',
        legend=False, grid=False, show=False)
    plt.savefig(f'vorticity_{name}_{beta:.2f}_{psi_max:.2e}.eps')
    plt.close()

    plot_utils.plot_value(
        utils.create_state_mtx(x, interface=interface)[:, :, 0, 4],
        interface, title=f'Salinity at $\\beta={beta:.2f}$ and $\\Psi_\\max={psi_max:.2e}$',
        legend=False, grid=False, show=False)
    plt.savefig(f'salinity_{name}_{beta:.2f}_{psi_max:.2e}.eps')
    plt.close()


def main():
    '''An example of performing a continuation for a 2D AMOC, where the plots and interim
    solutions are written to files instead of storing them in memory and showing them on
    the screen.'''

    nx = 40
    ny = 80

    # Define the problem
    parameters = {'Problem Type': 'AMOC',
                  # Problem parameters
                  'Rayleigh Number': 4e4,
                  'Prandtl Number': 2.25,
                  'Lewis Number': 1,
                  'Top Boundary Layer Thickness': 0.05,
                  'Beta': 0,
                  'Tau S': 1,
                  'Tau T': 0.1,
                  'X-max': 5,
                  # Give back extra output (this is also more expensive)
                  'Verbose': False}

    y = utils.create_stretched_coordinate_vector(0, 1, ny, 1.5)

    interface = Interface(parameters, nx, ny, y=y)
    continuation = Continuation(interface, newton_tolerance=1e-6)

    # First increase the temperature forcing to the desired value
    x0 = interface.vector()

    ds = 0.1
    target = 1
    x1 = continuation.continuation(x0, 'Temperature Forcing', 0, target, ds, ds_min=1e-8)[0]

    # Write the solution to a file
    interface.save_state('x1', x1)

    # Enable the lines below to load the solution instead. Same for the ones below
    # x1 = interface.load_state('x1')

    generate_plots(interface, x1, 'temperature', 1)

    ds = 0.1
    target = 1
    x2 = continuation.continuation(x1, 'Salinity Forcing', 0, target, ds, ds_min=1e-8)[0]

    # Write the solution to a file
    interface.save_state('x2', x2)

    generate_plots(interface, x2, 'salinity', 1)
    generate_plots(interface, x2, 'beta', 0)

    # Perform a continuation to beta 0.2 without detecting bifurcation points
    # and use this in the bifurcation diagram
    data3 = Data()

    ds = 0.05
    target = 0.2
    x3, beta3 = continuation.continuation(x2, 'Beta', 0, target,
                                          ds, ds_min=1e-12, callback=data3.callback)

    # Write the solution to a file
    interface.save_state('x3', x3)

    generate_plots(interface, x3, 'beta', beta3)

    # Write the data to a file
    with open('data3', 'wb') as f:
        pickle.dump(data3, f)

    # Plot a bifurcation diagram
    plt.title(f'Bifurcation diagram for the AMOC model with $n_x={nx}$, $n_y={ny}$')
    plt.xlabel('$\\beta$')
    plt.ylabel('Maximum value of the streamfunction')
    plt.plot(data3.mu, data3.value)
    plt.savefig('bifurcation_diagram.eps')
    plt.close()


if __name__ == '__main__':
    main()
