import deepxde as dde
import numpy as np
from deepxde.backend import tf
import scipy.io

dde.config.set_default_float("float64")
dde.config.disable_xla_jit()


def gen_data(num):

    data = scipy.io.loadmat("elastodynamics_fenics.mat")
    U = data["solid_disp"]
    S = data["solid_stress"]
    time = data["time"]
    XY_solid = data["solid_xy"]

    N_solid = XY_solid.shape[0]
    TS = time.shape[0]

    X = XY_solid[:, 0:1].flatten()[:, None]
    Y = XY_solid[:, 1:2].flatten()[:, None]

    count = 0
    for i in time:
        if i <= 0.21:
            if i == 0:
                XYT = np.hstack((X, Y, np.ones_like(X) * i))
                ux = U[:, 0, count].flatten()[:, None]
                uy = U[:, 1, count].flatten()[:, None]
                sxx = S[:, 0, count].flatten()[:, None]
                syy = S[:, 1, count].flatten()[:, None]
                sxy = S[:, 2, count].flatten()[:, None]
            elif count % 10 == 0:
                XYT_new = np.hstack((X, Y, np.ones_like(X) * i))
                XYT = np.vstack((XYT, XYT_new))
                ux_new = U[:, 0, count].flatten()[:, None]
                ux = np.vstack((ux, ux_new))
                uy_new = U[:, 1, count].flatten()[:, None]
                uy = np.vstack((uy, uy_new))
                sxx_new = S[:, 0, count].flatten()[:, None]
                sxx = np.vstack((sxx, sxx_new))
                syy_new = S[:, 1, count].flatten()[:, None]
                syy = np.vstack((syy, syy_new))
                sxy_new = S[:, 2, count].flatten()[:, None]
                sxy = np.vstack((sxy, sxy_new))
            count = count + 1

    num = int(num / 2)

    samplingRegion1 = XYT[:, 0] <= 1
    idx1 = np.where(samplingRegion1)[0]

    if idx1.shape[0] < num:
        idx1 = idx1
        num = int(num * 2 - idx1.shape[0])
    else:
        idx1 = np.random.choice(np.where(samplingRegion1)[0], num, replace=False)

    samplingRegion2 = XYT[:, 0] > 1
    idx2 = np.random.choice(np.where(samplingRegion2)[0], num, replace=False)

    nb = 25
    b1 = XYT[:, 0] == 10
    idx3 = np.random.choice(np.where(b1)[0], nb, replace=False)

    nb = 225
    b2 = XYT[:, 1] == 0
    idx4 = np.random.choice(np.where(b2)[0], nb, replace=False)

    b3 = XYT[:, 1] == 1
    idx5 = np.random.choice(np.where(b3)[0], nb, replace=False)

    XYT_star = np.vstack((XYT[idx1], XYT[idx2], XYT[idx3], XYT[idx4], XYT[idx5]))
    ux_star = np.vstack((ux[idx1], ux[idx2], ux[idx3], ux[idx4], ux[idx5]))
    uy_star = np.vstack((uy[idx1], uy[idx2], uy[idx3], uy[idx4], uy[idx5]))
    sxx_star = np.vstack((sxx[idx1], sxx[idx2], sxx[idx3], sxx[idx4], sxx[idx5]))
    syy_star = np.vstack((syy[idx1], syy[idx2], syy[idx3], syy[idx4], syy[idx5]))
    sxy_star = np.vstack((sxy[idx1], sxy[idx2], sxy[idx3], sxy[idx4], sxy[idx5]))

    return XYT_star, ux_star, uy_star, sxx_star, syy_star, sxy_star


def main():

    E_ = dde.Variable(1.0)
    nu_ = dde.Variable(1.0)
    rho = 1
    rho_g = 5
    observe_xyt, ux, uy, sxx, syy, sxy = gen_data(625)

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

        E = (tf.tanh(E_) + 1) * 5e6 / scale
        nu = (tf.tanh(nu_) + 1) / 4

        E = tf.cast(E, tf.float64)
        nu = tf.cast(nu, tf.float64)

        Cxx = E / ((1 + nu) * (1 - 2 * nu))
        Cyy = 1 - nu
        Cxy = E / (1 + nu)

        Sxx = Cxx * (Cyy * Exx + nu * Eyy)
        Syy = Cxx * (Cyy * Eyy + nu * Exx)
        Sxy = Cxy * Exy

        Sxx_x = dde.grad.jacobian(f, x, i=2, j=0)
        Syy_y = dde.grad.jacobian(f, x, i=3, j=1)
        Sxy_x = dde.grad.jacobian(f, x, i=4, j=0)
        Syx_y = dde.grad.jacobian(f, x, i=4, j=1)

        d2x_dt2 = dde.grad.hessian(Nux, x, i=2, j=2)
        d2y_dt2 = dde.grad.hessian(Nuy, x, i=2, j=2)

        Fx = 0
        Fy = -rho_g

        momentum_x = Sxx_x + Syx_y - Fx - rho * d2x_dt2
        momentum_y = Sxy_x + Syy_y - Fy - rho * d2y_dt2

        stress_x = Sxx - f[:, 2:3]
        stress_y = Syy - f[:, 3:4]
        stress_xy = Sxy - f[:, 4:5]

        return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]

    geom = dde.geometry.Rectangle([0, 0], [10, 1])
    timedomain = dde.geometry.TimeDomain(0, 0.2)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    scale = 1e3
    observe_ux = dde.PointSetBC(observe_xyt, ux * scale, component=0)
    observe_uy = dde.PointSetBC(observe_xyt, uy * scale, component=1)
    observe_sxx = dde.PointSetBC(observe_xyt, sxx, component=2)
    observe_syy = dde.PointSetBC(observe_xyt, syy, component=3)
    observe_sxy = dde.PointSetBC(observe_xyt, sxy, component=4)

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [observe_ux, observe_uy, observe_sxx, observe_syy, observe_sxy],
        num_domain=500,
        num_boundary=500,
        num_initial=100,
        num_test=500,
    )

    net = dde.nn.PFNN(
        [3, [20, 20, 20, 20, 20], [20, 20, 20, 20, 20], [20, 20, 20, 20, 20], 5],
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

        Nux = tf.where(tf.equal(x[:, 2:3], 0.0), tf.zeros_like(Nux), Nux)
        Nuy = tf.where(tf.equal(x[:, 2:3], 0.0), tf.zeros_like(Nuy), Nuy)
        Nsxx = tf.where(tf.equal(x[:, 2:3], 0.0), tf.zeros_like(Nsxx), Nsxx)
        Nsyy = tf.where(tf.equal(x[:, 2:3], 0.0), tf.zeros_like(Nsyy), Nsyy)
        Nsxy = tf.where(tf.equal(x[:, 2:3], 0.0), tf.zeros_like(Nsxy), Nsxy)

        return tf.concat(
            [
                Nux * np.max(np.abs(ux)) * scale,
                Nuy * np.max(np.abs(uy)) * scale,
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
        [E_, nu_], period=1000, filename="2D_dynamics.dat"
    )
    losshistory, train_state = model.train(epochs=1500000, callbacks=[variable])
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    return


if __name__ == "__main__":
    main()
