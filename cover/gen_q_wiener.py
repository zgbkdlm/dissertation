# Simulate an H^1_0([0, S])-valued Wiener process. This generates the cover image of the thesis.
# Zheng Zhao 2019 2021
#
# Reference: Gabriel J. Lord et al., 2014 spde book.
#
# note: Aalto platform does not support RGBA colour hence, cannot use alpha.
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

np.random.seed(1901)

# Paras
r = 2
J = 2 ** 7
K = J - 1

S = 2
xs = np.linspace(0, S, K)

dt = 5e-3
ts = np.arange(dt, 2 + dt, dt)

ps = np.arange(1, K + 1).reshape(1, -1) * 1.0
lam_j = ps ** (-(2 * r + 1))
sheet_jk = ps.T * ps

# Simulate Wiener processes
normal_incs = np.random.randn(ts.size, K)
dW = np.dot(np.sqrt(2 * lam_j * dt / S) * normal_incs / np.sqrt(dt), np.sin(np.pi * sheet_jk / J))
WW = np.cumsum(dW, 0)

# Plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

colours = cm.magma(np.linspace(0, 0.9, ts.shape[0]))
for t, Wt, colour in zip(ts, WW, colours):
    _, = ax.plot3D(xs, [t] * K, Wt, linewidth=0.1, color=colour, alpha=1.)

ax.grid(False)

ax.set_axis_off()

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# Transparent spines
ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False

ax.view_init(23, 25)

bbox = fig.bbox_inches.from_bounds(1.91, 1.48, 2.775, 1.843)

# Save in pdf and png
png_meta_data = {'Title': 'A Q-Wiener process realisation',
                 'Author': 'Zheng Zhao',
                 'Copyright': 'Zheng Zhao',
                 'Description': 'https://github.com/zgbkdlm/dissertation'}
pdf_meta_data = {'Title': 'A Q-Wiener process realisation',
                 'Author': 'Zheng Zhao'}
plt.savefig('cover.png', dpi=1200, transparent=True, metadata=png_meta_data, bbox_inches=bbox)
plt.savefig('cover.pdf', bbox_inches=bbox, metadata=pdf_meta_data)

