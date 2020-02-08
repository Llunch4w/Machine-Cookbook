

import numpy as np
import matplotlib as mpl
# mpl.use("TkAgg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animmation

r1 = 10
r2 = 1
omega1 = 2 * np.pi
omega2 = 24 * np.pi 
phi = 5.1454 * np.pi / 180 

def update(data):
    global line1, line2 , line3
    line1.set_data([data[0], data[1]])
    line1.set_3d_properties(data[2])
    line2.set_data([data[3], data[4]])
    line2.set_3d_properties(data[5])
    line3.set_data([data[6], data[7]])
    line3.set_3d_properties(data[8])
    return line1,line2,line3,
def init():
    global line1, line2, line3
    ti = 0
    t = t_drange[np.mod(ti, t_dlen)]

    xt1 = x0 + r1 * np.cos(omega1 * t)
    yt1 = y0 + r1 * np.sin(omega1 * t)
    zt1 = z0 + 0
    xt2 = xt1 + r2 * np.sin(omega2 * t)
    yt2 = yt1 + r2 * np.cos(omega2 * t)/(np.cos(phi) * (1 + np.tan(phi) ** 2))
    zt2 = zt1 + (yt2 - yt1) * np.tan(phi)
    xt21 = xt1 + r2 * np.sin(2 * np.pi * t_range)
    yt21 = yt1 + r2 * np.cos(2 * np.pi * t_range)/(np.cos(phi) * (1 + np.tan(phi) ** 2))
    zt21 = zt1 + (yt21 - yt1) * np.tan(phi)


    line1, = ax.plot([xt1], [yt1], [zt1], marker='o', color='blue',markersize=8)
    line2, = ax.plot([xt2], [yt2], [zt2], marker='o', color='orange',markersize=4)
    line3, = ax.plot(xt21, yt21, zt21, color='purple')
    return line1,line2,line3
def data_gen():
    #global x0,y0,z0,ti, t_drang, t_range, omega1, omega2, phi
    global x0,y0,z0,t_dlen

    #while true:
    data = []
    for ti in range(1,t_dlen):
        t = t_drange[ti]
        xt1 = x0 + r1 * np.cos(omega1 * t)
        yt1 = y0 + r1 * np.sin(omega1 * t)
        zt1 = z0
        xt2 = xt1 + r2 * np.sin(omega2 * t)
        yt2 = yt1 + r2 * np.cos(omega2 * t)/(np.cos(phi) * (1 + np.tan(phi) ** 2))
        zt2 = zt1 + (yt2 - yt1) * np.tan(phi)
        xt21 = xt1 + r2 * np.sin(2 * np.pi * t_range)
        yt21 = yt1 + r2 * np.cos(2 * np.pi * t_range)/(np.cos(phi) * (1 + np.tan(phi) ** 2))
        zt21 = zt1 + (yt21 - yt1) * np.tan(phi)
        data.append([xt1, yt1, zt1, xt2, yt2, zt2, xt21, yt21, zt21])
    return data
    #yield (xt1, yt1, zt1, xt2, yt2, zt2, xt21, yt21, zt21)


t_range = np.arange(0, 1 + 0.005, 0.005)
t_drange = np.arange(0, 1, 0.005 )
t_len = len(t_range)
t_dlen = len(t_drange)
#sun's coordination
x0 = 0
y0 = 0 
z0 = 0
#earth's orbit
x1 = x0  + r1 * np.cos(omega1 * t_range)
y1 = y0 + r1 * np.sin(omega1 * t_range)
z1 = z0 + np.zeros(t_len)

#moon's orbit
x2 = x1 + r2 * np.sin(omega2 * t_range)
y2 = y1 + r2 * np.cos(omega2 * t_range)/(np.cos(phi) * (1 + np.tan(phi) ** 2))
z2 = z1 + (y2 - y1) * np.tan(phi)


f = plt.figure(figsize=(6,6))
ax  = f.add_subplot(111,projection='3d')
#plt.rcParams['animation.ffmpeg_path'] = r"C:\Program Files\ffmpeg\bin\ffmpeg"
#plt.rcParams['animation.convert_path'] = r"C:\Program Files\ImageMagick-7.0.7-Q16\magick.exe"
ax.set_aspect('equal')
ax.set_title("Sun-Earth-Moon Model")

ax.plot([0], [0], [0], marker='o', color= 'red', markersize=16)
ax.plot(x1, y1, z1, 'r')
ax.plot(x2, y2, z2, 'b')
ax.set_xlim([-(r1 + 2), (r1 + 2)])
ax.set_ylim([-(r1 + 2), (r1 + 2)])
ax.set_zlim([-5, 5])
# line1 update Earth's track  dynamically
# line2 update Moon's track dynamically
# line3 update Moon's orbit to earth 
line1, = ax.plot([], [], [], marker='o', color='blue',markersize=8,animated = True)
line2, = ax.plot([], [], [], marker='o', color='orange',markersize=4,animated = True)
line3, = ax.plot([], [], [], color='purple',animated = True)

#red sphere for Sun, blue sphere for Earth, orange sphere for Moon
ani = animmation.FuncAnimation(f, update, frames = data_gen(), init_func = init,interval = 20)
#ffwriter = animmation.ffmpegwriter(fps = 200)
#ani.save('planet.gif', writer='imagemagick', fps=40)
#ani.save('planet.gif', writer = ffwriter)
plt.show()
