import deepxde as dde
import numpy as np
import pandas as pd
from deepxde.backend import tf

dde.config.set_default_float("float64")
dde.config.disable_xla_jit()


def gen_data_m1(num):

    data = pd.read_csv("FEA/neoHookeanDisp_fea_m1.csv")
    X = data["x"].values.flatten()[:, None]
    Y = data["y"].values.flatten()[:, None]
    ux = data["ux"].values.flatten()[:, None]
    uy = data["uy"].values.flatten()[:, None]

    data = pd.read_csv("FEA/neoHookeanCauchyStress_fea_m1.csv")
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
    if idx1.shape[0] < num:
        idx1 = idx1
        num = int(num * 2 - idx1.shape[0])
    else:
        idx1 = np.random.choice(np.where(samplingRegion1)[0], num, replace=False)

    samplingRegion2 = X_star[:, 0] > 1
    idx2 = np.random.choice(np.where(samplingRegion2)[0], num, replace=False)

    nb = 10
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


def gen_data_m2():

    data = pd.read_csv("FEA/neoHookeanDisp_fea_m2.csv")
    X = data["x"].values.flatten()[:, None]
    Y = data["y"].values.flatten()[:, None]
    ux = data["ux"].values.flatten()[:, None]
    uy = data["uy"].values.flatten()[:, None]

    data = pd.read_csv("FEA/neoHookeanCauchyStress_fea_m2.csv")
    sxx = data["sxx"].values.flatten()[:, None]
    syy = data["syy"].values.flatten()[:, None]
    sxy = data["sxy"].values.flatten()[:, None]
    syx = data["sxy"].values.flatten()[:, None]

    XY_star = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    ux_star = ux.flatten()[:, None]
    uy_star = uy.flatten()[:, None]
    sxx_star = sxx.flatten()[:, None]
    syy_star = syy.flatten()[:, None]
    sxy_star = sxy.flatten()[:, None]

    return XY_star, ux_star, uy_star, sxx_star, syy_star, sxy_star


def gen_data_m3():

    data = pd.read_csv("FEA/neoHookeanDisp_fea_m3.csv")
    X = data["x"].values.flatten()[:, None]
    Y = data["y"].values.flatten()[:, None]
    ux = data["ux"].values.flatten()[:, None]
    uy = data["uy"].values.flatten()[:, None]

    data = pd.read_csv("FEA/neoHookeanCauchyStress_fea_m3.csv")
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

    ind1 = np.where((X_star[:, 0] == 0))[0]
    ind2 = np.where((X_star[:, 0] == 10))[0]
    ind3 = np.where((X_star[:, 1] == 0))[0]
    ind4 = np.where((X_star[:, 1] == 1))[0]

    XY_star = np.vstack((X_star[ind1], X_star[ind2], X_star[ind3], X_star[ind4]))
    ux_star = np.vstack((ux[ind1], ux[ind2], ux[ind3], ux[ind4]))
    uy_star = np.vstack((uy[ind1], uy[ind2], uy[ind3], uy[ind4]))
    sxx_star = np.vstack((sxx[ind1], sxx[ind2], sxx[ind3], sxx[ind4]))
    syy_star = np.vstack((syy[ind1], syy[ind2], syy[ind3], syy[ind4]))
    sxy_star = np.vstack((sxy[ind1], sxy[ind2], sxy[ind3], sxy[ind4]))

    return XY_star, ux_star, uy_star, sxx_star, syy_star, sxy_star


def main():

    E_ = dde.Variable(1.0)
    nu_ = dde.Variable(1.0)
    rho_g = 0.1
    observe_xy, ux, uy, sxx, syy, sxy = gen_data_m1(250)
    # observe_xy, ux, uy, sxx, syy, sxy = gen_data_m2()
    # observe_xy, ux, uy, sxx, syy, sxy = gen_data_m3()

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

        duxdx = dde.grad.jacobian(Nux, x, i=0, j=0)
        duydy = dde.grad.jacobian(Nuy, x, i=0, j=1)
        duxdy = dde.grad.jacobian(Nux, x, i=0, j=1)
        duydx = dde.grad.jacobian(Nuy, x, i=0, j=0)

        Fxx = duxdx + 1
        Fxy = duxdy
        Fyx = duydx
        Fyy = duydy + 1

        detF = Fxx * Fyy - Fxy * Fyx

        invFxx = Fyy / detF
        invFyy = Fxx / detF
        invFxy = -Fxy / detF
        invFyx = -Fyx / detF

        E = (tf.tanh(E_) + 1.0) * 2e4  # bound [0, 4]e4
        nu = (tf.tanh(nu_) + 1.0) / 4  # bound [0, 0.5]

        E = tf.cast(E, tf.float64)
        nu = tf.cast(nu, tf.float64)

        lmbd = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        mu = tf.cast(mu, tf.float64)
        lmbd = tf.cast(lmbd, tf.float64)

        # compressible 1st PK Stress(incompressible 1st PK stress: P = -pF^(-T)+muF)
        Pxx = mu * Fxx + (lmbd * tf.math.log(detF) - mu) * invFxx
        Pxy = mu * Fxy + (lmbd * tf.math.log(detF) - mu) * invFyx
        Pyx = mu * Fyx + (lmbd * tf.math.log(detF) - mu) * invFxy
        Pyy = mu * Fyy + (lmbd * tf.math.log(detF) - mu) * invFyy

        # Cauchy stress
        Sxx = (Pxx * Fxx + Pxy * Fxy) / detF
        Syx = (Pyx * Fxx + Pyy * Fxy) / detF
        Sxy = (Pxx * Fyx + Pxy * Fyy) / detF
        Syy = (Pyx * Fyx + Pyy * Fyy) / detF

        Sxx_x = dde.grad.jacobian(f, x, i=2, j=0)
        Syy_y = dde.grad.jacobian(f, x, i=3, j=1)
        Syx_y = dde.grad.jacobian(f, x, i=4, j=1)
        Sxy_x = dde.grad.jacobian(f, x, i=4, j=0)

        Fx = 0
        Fy = -rho_g

        momentum_x = Sxx_x + Syx_y - Fx
        momentum_y = Sxy_x + Syy_y - Fy

        stress_x = Sxx - f[:, 2:3]
        stress_y = Syy - f[:, 3:4]
        stress_xy = Sxy - f[:, 4:5]

        return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]

    geom = dde.geometry.Rectangle([0, 0], [10, 1])

    observe_ux = dde.PointSetBC(observe_xy, ux, component=0)
    observe_uy = dde.PointSetBC(observe_xy, uy, component=1)
    observe_sxx = dde.PointSetBC(observe_xy, sxx, component=2)
    observe_syy = dde.PointSetBC(observe_xy, syy, component=3)
    observe_sxy = dde.PointSetBC(observe_xy, sxy, component=4)

    data = dde.data.PDE(
        geom,
        pde,
        [observe_ux, observe_uy, observe_sxx, observe_syy, observe_sxy],
        num_domain=200,
        num_boundary=200,
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

        Nux = tf.where(tf.equal(x[:, 0:1], 0.0), tf.zeros_like(Nux), Nux)
        Nuy = tf.where(tf.equal(x[:, 0:1], 0.0), tf.zeros_like(Nuy), Nuy)

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
        "adam", lr=1e-3, loss_weights=[1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1, 1, 1, 1, 1]
    )
    variable = dde.callbacks.VariableValue(
        [E_, nu_], period=1000, filename="2D_neohookean_static_variables.dat"
    )
    losshistory, train_state = model.train(epochs=1000000, callbacks=[variable])
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    return


if __name__ == "__main__":
    main()
