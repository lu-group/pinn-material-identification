import deepxde as dde
import numpy as np
from deepxde.backend import tf
import pandas as pd

dde.config.set_default_float("float64")
dde.config.disable_xla_jit()


def gen_data(num):

    data = pd.read_csv("linearElasticDisp_fea.csv")
    X = data["x"].values.flatten()[:, None]
    Y = data["y"].values.flatten()[:, None]
    ux = data["ux"].values.flatten()[:, None]
    uy = data["uy"].values.flatten()[:, None]

    data = pd.read_csv("linearElasticCauchyStress_fea.csv")
    sxx = data["sxx"].values.flatten()[:, None]
    syy = data["syy"].values.flatten()[:, None]
    sxy = data["sxy"].values.flatten()[:, None]
    syx = data["sxy"].values.flatten()[:, None]

    X_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    ux = ux.flatten()[:, None]
    uy = uy.flatten()[:, None]
    sxx = sxx.flatten()[:, None]
    syy = syy.flatten()[:, None]
    sxy = sxy.flatten()[:, None]

    num = int(num / 2)

    samplingRegion1 = X_star[:, 0] <= 1
    idx1 = np.where(samplingRegion1)[0]
    # print(idx1.shape[0])
    if idx1.shape[0] < num:
        idx1 = idx1
        num = int(num * 2 - idx1.shape[0])
    else:
        idx1 = np.random.choice(np.where(samplingRegion1)[0], num, replace=False)

    samplingRegion2 = X_star[:, 0] > 1
    idx2 = np.random.choice(np.where(samplingRegion2)[0], num, replace=False)

    nb = 11
    b1 = X_star[:, 0] == 10
    idx3 = np.random.choice(np.where(b1)[0], nb, replace=False)

    nb = 90
    b2 = X_star[:, 1] == 0
    idx4 = np.random.choice(np.where(b2)[0], nb, replace=False)

    b3 = X_star[:, 1] == 1
    idx5 = np.random.choice(np.where(b3)[0], nb, replace=False)

    XY_star = np.vstack(
        (X_star[idx1], X_star[idx2], X_star[idx3], X_star[idx4], X_star[idx5])
    )
    ux_star = np.vstack((ux[idx1], ux[idx2], ux[idx3], ux[idx4], ux[idx5]))
    uy_star = np.vstack((uy[idx1], uy[idx2], uy[idx3], uy[idx4], uy[idx5]))
    sxx_star = np.vstack((sxx[idx1], sxx[idx2], sxx[idx3], sxx[idx4], sxx[idx5]))
    syy_star = np.vstack((syy[idx1], syy[idx2], syy[idx3], syy[idx4], syy[idx5]))
    sxy_star = np.vstack((sxy[idx1], sxy[idx2], sxy[idx3], sxy[idx4], sxy[idx5]))

    return XY_star, ux_star, uy_star, sxx_star, syy_star, sxy_star


def main():

    E_ = dde.Variable(1.0)
    nu_ = dde.Variable(1.0)
    rho_g = 1
    observe_xy, ux, uy, sxx, syy, sxy = gen_data(250)

    def pde(x, f):
        """
        x: Network input
        x[:,0] is the x-coordinate
        x[:,1] is the y-coordinate
        f: Network output
        f[:,0] is Nux
        f[:,1] is Nuy
        f[:,2] is Nsxx
        f[:,3] is Nsyy
        f[:,4] is Nsxy
        """
        Nux, Nuy = f[:, 0:1], f[:, 1:2]

        Exx = dde.grad.jacobian(Nux, x, i=0, j=0)
        Eyy = dde.grad.jacobian(Nuy, x, i=0, j=1)
        Exy = 0.5 * (
            dde.grad.jacobian(Nux, x, i=0, j=1) + dde.grad.jacobian(Nuy, x, i=0, j=0)
        )

        E = (tf.tanh(E_) + 1.0) * 3e5
        nu = (tf.tanh(nu_) + 1.0) / 4

        E = tf.cast(E, tf.float64)
        nu = tf.cast(nu, tf.float64)

        Sxx = E / (1 - nu ** 2) * (Exx + nu * Eyy)
        Syy = E / (1 - nu ** 2) * (Eyy + nu * Exx)
        Sxy = E / (1 + nu) * Exy

        Sxx_x = dde.grad.jacobian(f, x, i=2, j=0)
        Syy_y = dde.grad.jacobian(f, x, i=3, j=1)
        Sxy_x = dde.grad.jacobian(f, x, i=4, j=0)
        Syx_y = dde.grad.jacobian(f, x, i=4, j=1)

        Fx = 0
        Fy = -rho_g

        momentum_x = Sxx_x + Syx_y - Fx
        momentum_y = Sxy_x + Syy_y - Fy

        stress_x = Sxx - f[:, 2:3]
        stress_y = Syy - f[:, 3:4]
        stress_xy = Sxy - f[:, 4:5]

        return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]

    geom = dde.geometry.Rectangle([0, 0], [10, 1])

    def left_boundary(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0)

    bc_l1 = dde.DirichletBC(geom, lambda x: 0, left_boundary, component=0)
    bc_l2 = dde.DirichletBC(geom, lambda x: 0, left_boundary, component=1)

    observe_ux = dde.PointSetBC(observe_xy, ux, component=0)
    observe_uy = dde.PointSetBC(observe_xy, uy, component=1)
    observe_sxx = dde.PointSetBC(observe_xy, sxx, component=2)
    observe_syy = dde.PointSetBC(observe_xy, syy, component=3)
    observe_sxy = dde.PointSetBC(observe_xy, sxy, component=4)

    data = dde.data.PDE(
        geom,
        pde,
        [bc_l1, bc_l2, observe_ux, observe_uy, observe_sxx, observe_syy, observe_sxy],
        num_domain=100,
        num_boundary=50,
        num_test=100,
    )

    net = dde.nn.PFNN(
        [2, [15, 15, 15, 15, 15], [15, 15, 15, 15, 15], [15, 15, 15, 15, 15], 5],
        "tanh",
        "Glorot normal",
    )

    def output_transform(x, f):

        Nux, Nuy, Nsxx, Nsyy, Nsxy = (
            f[:, 0:1],
            f[:, 1:2],
            f[:, 2:3],
            f[:, 3:4],
            f[:, 4:5],
        )

        return tf.concat(
            [
                Nux * np.max(np.abs(ux)),
                Nuy * np.max(np.abs(uy)),
                Nsxx * np.max(np.abs(sxx)),
                Nsyy * np.max(np.abs(syy)),
                Nsxy * np.max(np.abs(sxy)),
            ],
            axis=1,
        )

    net.apply_output_transform(output_transform)
    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=1e-3,
        loss_weights=[1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1, 1, 1, 1, 1, 1, 1],
    )
    variable = dde.callbacks.VariableValue(
        [E_, nu_], period=1000, filename="2D_elastic_static_variables.dat"
    )
    losshistory, train_state = model.train(epochs=1000000, callbacks=[variable])
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    return


if __name__ == "__main__":
    main()
