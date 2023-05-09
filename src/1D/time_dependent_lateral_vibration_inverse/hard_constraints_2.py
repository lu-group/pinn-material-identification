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
    u = np.sin(pi * XT[:, 0:1]) * np.cos(pi ** 2 * XT[:, 1:2])

    return XT, u


def main():

    lmbd_ = dde.Variable(1.0)
    observe_xt, u_exact = gen_data()

    def pde(x, f):
        """
        x: Network input
        x[:,0] is the x-coordinate
        x[:,1] is the time
        f: Network output
        f[:,0] is Nu
        """
        d2u_dt2 = dde.grad.hessian(f, x, i=1, j=1)
        d2u_dx2 = dde.grad.hessian(f, x, i=0, j=0)
        d4u_dx4 = dde.grad.hessian(d2u_dx2, x, i=0, j=0)

        lmbd = (tf.tanh(lmbd_) + 1) * 2
        equilibrium = d2u_dt2 + lmbd * d4u_dx4

        return [equilibrium]

    geom = dde.geometry.Interval(0.0, 1.0)
    timedomain = dde.geometry.TimeDomain(0.0, 1.0)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    observe_u = dde.icbc.PointSetBC(observe_xt, u_exact, component=0)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [observe_u],
        num_domain=100,
        num_boundary=50,
        num_initial=50,
        num_test=50,
    )

    net = dde.nn.FNN([2] + 3 * [50] + [1], "tanh", "Glorot normal")
    net.apply_output_transform(
        lambda x, y: tf.sin(np.pi * x[:, 0:1]) * x[:, 1:2] * x[:, 1:2] * y
        + tf.sin(np.pi * x[:, 0:1]) * tf.cos(3 * np.pi / 2 * x[:, 1:2])
    )

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, external_trainable_variables=[lmbd_])
    variable = dde.callbacks.VariableValue(
        [lmbd_], period=1000, filename="lateral_variables.dat"
    )
    losshistory, train_state = model.train(epochs=1000000, callbacks=[variable])
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    u_pred = model.predict(observe_xt)
    l2_difference = dde.metrics.l2_relative_error(u_exact, u_pred)

    print("L2 relative error in u:", l2_difference)

    return


if __name__ == "__main__":
    main()
