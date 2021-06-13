import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import pickle
import os
import sys
from datetime import datetime

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def date_string():
    return datetime.now().strftime("%Y-%m-%d|%H.%M.%S")


def create_dir_from_path_if_not_exists(path):
    path = os.path.dirname(path)
    if not os.path.exists(path) and path != '':
        os.makedirs(path)


def pdump(obj, file_name):
    create_dir_from_path_if_not_exists(file_name)

    po = open(f"{file_name}", "wb")
    pickle.dump(obj, po)
    po.close()


def pload(file_name):
    po = open(f"{file_name}", "rb")
    obj = pickle.load(po)
    po.close()
    return obj


def basic_logger(logger_path):
    log_dir, _ = os.path.split(logger_path)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(logger_path)

    logger.setLevel(logging.INFO)
    format_string = "%(asctime)s [%(levelname)s]: %(message)s"
    log_format = logging.Formatter(format_string, "%Y-%m-%d %H:%M:%S")

    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_path)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger


##########
# Shapes #
##########

def circle(num_landmarks, scale=1, shift=0):
    thetas = np.linspace(0, 2 * np.pi, num=num_landmarks + 1)[:-1]
    positions = scale * np.array([[np.cos(x), np.sin(x)] for x in thetas])
    return positions + np.array([shift, shift])


def triangle(num_landmarks):
    if num_landmarks % 3 != 0:
        raise Exception("Want a nice image, so satisfy 'num_landmarks % 3 == 0' !")

    a = np.array([1e-06, .3])  # lazy with sign(0) in reflection
    b = np.array([-.3, 0])
    c = np.array([.3, 0])

    # interpolate between them to generate points
    ss = num_landmarks // 3
    ss0 = 1
    ss1 = 4
    q0_a = [(1 - s / ss0) * a + s / ss0 * b for s in range(ss)]  # [`a` -> `b`)
    q0_b = [(1 - s / ss1) * b + s / ss1 * c for s in range(ss)]  # [`b` -> `c`)
    q0_c = [(1 - s / ss0) * c + s / ss0 * a for s in range(ss)]  # [`c` -> `a`)

    return np.array(q0_a + q0_b + q0_c)


def rectangle(num_landmarks):
    if num_landmarks % 4 != 0:
        raise Exception("Want a nice image, so satisfy 'num_landmarks % 4 == 0' !")

    a = np.array([1, 1])
    b = np.array([2, 1])
    c = np.array([2, 2])
    d = np.array([1, 2])

    # interpolate between them to generate points
    ss = num_landmarks // 4
    pts = lambda x, y: [(1 - s / ss) * x + s / ss * y for s in range(ss)]
    return np.array(pts(a, b) + pts(b, c) + pts(c, d) + pts(d, a))


############
# Plotting #
############

def plot_landmarks(file_name,
                   qs=None,
                   template=None,
                   target=None,
                   title=None,
                   template_label='$\mathbf{q}_0$',
                   target_label='$\mathbf{q}_1$',
                   landmark_label='$\mathbf{q}_{\{0<t<1\}}$'
                   ):
    # TODO: fix this mess

    create_dir_from_path_if_not_exists(file_name)
    if isinstance(qs, torch.Tensor):
        qs = qs.detach().numpy()

    plt.figure(figsize=(5, 4))
    if qs is None:
        assert template is not None and target is not None
        template = template.detach().numpy()
        q0_ext = np.vstack((template, template[0, :]))
        plt.plot(q0_ext[:, 0], q0_ext[:, 1], c='b', marker='o', label=template_label, zorder=2)
        target = target.detach().numpy()
        q1_ext = np.vstack((target, target[0, :]))
        plt.plot(q1_ext[:, 0], q1_ext[:, 1], c='r', marker='x', label=target_label, zorder=2)
    elif len(qs.shape) == 2:
        qs_ext = np.vstack((qs, qs[0, :]))  # needs improving
        plt.plot(qs_ext[:, 0], qs_ext[:, 1], c='k', lw=0.75, zorder=1, label=landmark_label)
        if template is not None:
            template = template.detach().numpy()
            q0_ext = np.vstack((template, template[0, :]))
            plt.plot(q0_ext[:, 0], q0_ext[:, 1], c='b', marker='o', label=template_label, zorder=2)
        if target is not None:
            target = target.detach().numpy()
            q1_ext = np.vstack((target, target[0, :]))
            plt.plot(q1_ext[:, 0], q1_ext[:, 1], c='r', marker='x', label=target_label, zorder=2)
    elif len(qs.shape) == 3:
        for i in range(len(qs[0]) - 1):
            plt.plot(qs[:, i, 0], qs[:, i, 1], c='k', lw=0.75, zorder=1)
        # plot last one with label
        plt.plot(qs[:, -1, 0], qs[:, -1, 1], c='k', lw=0.75, zorder=1, label=landmark_label)

        template, target = qs[0, :, :], qs[-1, :, :]
        q0_ext, q1_ext = np.vstack((template, template[0, :])), np.vstack((target, target[0, :]))
        plt.plot(q0_ext[:, 0], q0_ext[:, 1], c='b', marker='o', label=template_label, zorder=2)
        plt.plot(q1_ext[:, 0], q1_ext[:, 1], c='r', marker='x', label=target_label, zorder=2)
    else:
        raise ValueError("Dimensions wrong.")
    if title:
        plt.title(title)

    plt.legend(loc='best', fontsize=14)
    plt.grid(linestyle='dotted')
    plt.ylabel('y', fontsize=20)
    plt.xlabel('x', fontsize=20)

    plt.savefig(file_name + '.pdf', bbox_inches='tight')
    plt.close()


def plot_misfits(misfits_path, file_name):
    plt.figure()
    po = open(misfits_path, 'rb')
    misfits = pickle.load(po)
    po.close()

    # plot log-data misfit
    plt.semilogy(range(len(misfits)), misfits)
    plt.xlabel(r'Iteration $k$')
    plt.ylabel(r'$\log E^k$')
    plt.grid(linestyle='dotted')

    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


def plot_consensus(consensus_path, file_name):
    po = open(consensus_path, "rb")
    consensus = pickle.load(po)
    po.close()

    plt.figure()
    plt.semilogy(range(len(consensus)), consensus)

    plt.xlabel(r'Iteration $k$')
    plt.ylabel(r'$\mathcal{S}^k$')
    plt.grid(linestyle='dotted')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


####################################
# Probability densities & sampling #
####################################

def pdf_vonMises(x, kappa, mu):
    from scipy.special import j0
    return np.exp(kappa * np.cos(x - mu)) / (2 * np.pi * j0(kappa))


def inverse_pdf_vonMises(u, kappa):
    """ Inverse PDF for the von Mises distribution with mu=0. """
    if np.fabs(kappa) < 1e-10:
        raise ValueError("Kappa too small.")
    np.seterr(all='raise')
    inv_kappa = 1. / kappa
    return 1 + inv_kappa * np.log(u + (1 - u) * np.exp(-2 * kappa))


def sample_vonMises(shape, kappa):
    """ Returns `num` von Mises distributed numbers on S^1 """
    us = np.random.uniform(size=shape)
    return inverse_pdf_vonMises(us, kappa)


def sample_normal(size, mu=0, std=1):
    dim = 2
    _mu = mu * np.ones(shape=dim)
    _std = std * np.ones(shape=(dim, dim))
    return np.random.normal(size=(size, dim), loc=mu, scale=std)
