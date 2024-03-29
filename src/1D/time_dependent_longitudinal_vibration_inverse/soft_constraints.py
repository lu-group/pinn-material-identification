import deepxde as dde
import numpy as np
from deepxde.backend import tf
import math

pi = math.pi


def gen_data():

    num = 100
    x = np.linspace(0, 1, num)
    t = np.linspace(0, 1, num)
    X, T = np.meshgrid(x, t)

    xt = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    nb, nd = 40, 500

    b1 = xt[:, 0] == 0
    idx1 = np.random.choice(np.where(b1)[0], nb, replace=False)
    b2 = xt[:, 0] == 1
    idx2 = np.random.choice(np.where(b2)[0], nb, replace=False)
    b3 = xt[:, 1] == 0
    idx3 = np.random.choice(np.where(b3)[0], nb, replace=False)
    b4 = xt[:, 1] == 1
    idx4 = np.random.choice(np.where(b4)[0], nb, replace=False)

    bc_idx = np.hstack((idx1, idx2, idx3, idx4))
    bc_total = np.hstack((b1, b2, b3, b4))

    interior = np.setdiff1d(np.arange(0, 10000, 1), bc_total)
    idx5 = np.random.choice(np.where(interior)[0], nd, replace=False)

    XT = np.vstack((xt[bc_idx], xt[idx5]))

    # Exact solution
    u = np.sin(pi * XT[:, 0:1]) * np.cos(pi * XT[:, 1:2])

    return XT, u


def main():

    alpha = dde.Variable(1.0)
    observe_xt, u_exact = gen_data()

    def pde(x, f):
        """
        x: Network input
        x[:,0] is the x-coordinate
        x[:,1] is the time
        f: Network output
        f[:,0] is u
        """
        d2u_dx2 = dde.grad.hessian(f, x, i=0, j=0)
        d2u_dt2 = dde.grad.hessian(f, x, i=1, j=1)

        equilibrium = d2u_dt2 - alpha * d2u_dx2

        return [equilibrium]

    geom = dde.geometry.Interval(0.0, 1.0)
    timedomain = dde.geometry.TimeDomain(0.0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def left_boundary(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0.0)

    def right_boundary(x, on_boundary):
        return on_boundary and np.isclose(x[0], 1.0)

    bc_l1 = dde.DirichletBC(geomtime, lambda x: 0, left_boundary, component=0)
    bc_r1 = dde.DirichletBC(geomtime, lambda x: 0, right_boundary, component=0)
    ic1 = dde.IC(
        geomtime,
        lambda x: np.sin(pi * x[:, 0:1]),
        lambda _, on_initial: on_initial,
        component=0,
    )
    ic2 = dde.OperatorBC(
        geomtime,
        lambda x, y, _: dde.grad.jacobian(y, x, i=0, j=1),
        lambda x, _: np.isclose(x[1], 0),
    )
    observe_u = dde.icbc.PointSetBC(observe_xt, u_exact, component=0)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc_l1, bc_r1, ic1, ic2, observe_u],
        num_domain=20,
        num_boundary=10,
        num_initial=10,
        num_test=10,
    )

    net = dde.nn.FNN([2] + 3 * [50] + [1], "tanh", "Glorot normal")

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, external_trainable_variables=[alpha])
    variable = dde.callbacks.VariableValue(
        [alpha], period=1000, filename="longitudinal_variables.dat"
    )
    losshistory, train_state = model.train(epochs=1000000, callbacks=[variable])
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    u_pred = model.predict(observe_xt)
    l2_difference = dde.metrics.l2_relative_error(u_exact, u_pred)

    print("L2 relative error in u:", l2_difference)

    return


if __name__ == "__main__":
    main()
