from __future__ import division
from math import *
import numpy as np
from datetime import datetime,timedelta
import matplotlib.pyplot as plt

def calculate_julian_century(timestamp):
    """
    Take a datetime object and returns the decimal
    Julian century.
    """
    origin_date = datetime(1899,12,31,12,00,00)# Noon Dec 31, 1899
    dt = timestamp - origin_date
    days = dt.days + dt.seconds/3600./24.
    return days/36525, timestamp.hour + timestamp.minute/60.

def solve_longman(lat,lon,alt,time):

    #T = JDate/36525
    #t0 = (JDate-floor(J0)-0.5)*24.

    T,t0 = calculate_julian_century(time)

    if t0 < 0:
        t0 += 24.
    # Not sure if this is needed, but it makes me happy
    if t0 >= 24:
        t0 -= 24.
    #print T,t0

    mu = 6.673e-8
    M = 7.3537e25
    S = 1.993e33
    e = 0.05490
    m = 0.074804
    c = 3.84402e10
    c1 = 1.495e13
    h2=0.612
    k2=0.303
    i = 0.08979719
    omega = radians(23.452)

    L = -1*lon # for some reason his lat/lon is - to ours
    lamb = radians(lat)
    a = 6.378270e8
    H = alt

    s = 4.72000889397 + 8399.70927456 * T + 3.45575191895e-05 * T * T + 3.49065850399e-08 * T * T * T
    p = 5.83515162814 + 71.0180412089 * T + 0.000180108282532 * T * T + 1.74532925199e-07 * T * T * T
    h = 4.88162798259 + 628.331950894 * T + 5.23598775598e-06 * T * T
    N = 4.52360161181 - 33.757146295 * T + 3.6264063347e-05 * T * T +  3.39369576777e-08 * T * T * T
    I = acos(cos(omega)*cos(i) - sin(omega)*sin(i)*cos(N))
    nu = asin(sin(i)*sin(N)/sin(I))
    t = radians(15. * (t0 - 12) - L)


    chi = t + h - nu
    cos_alpha = cos(N)*cos(nu)+sin(N)*sin(nu)*cos(omega)
    sin_alpha = sin(omega)*sin(N)/sin(I)
    alpha = 2*atan(sin_alpha/(1+cos_alpha))
    xi = N-alpha

    sigma = s - xi
    l = sigma + 2*e*sin(s-p)+(5./4)*e*e*sin(2*(s-p)) + (15./4)*m*e*sin(s-2*h+p) + (11./8)*m*m*sin(2*(s-h))

    # Sun
    p1 = 4.90822941839 + 0.0300025492114 * T +  7.85398163397e-06 * T * T + 5.3329504922e-08 * T * T * T
    e1 = 0.01675104-0.00004180*T - 0.000000126*T*T
    chi1 = t+h
    l1 = h + 2*e1*sin(h-p1)
    cos_theta = sin(lamb)*sin(I)*sin(l) + cos(lamb)*(cos(0.5*I)**2 * cos(l-chi) + sin(0.5*I)**2 * cos(l+chi))
    cos_phi = sin(lamb)*sin(omega)*sin(l1) + cos(lamb)*(cos(0.5*omega)**2 * cos(l1-chi1)+sin(0.5*omega)**2*cos(l1+chi1))

    # Distance
    C = sqrt(1./(1+0.006738*sin(lamb)**2))
    r = C*a + H
    aprime = 1./(c*(1-e*e))
    aprime1 = 1./(c1*(1-e1*e1))
    d = 1./((1./c) + aprime*e*cos(s-p)+aprime*e*e*cos(2*(s-p)) + (15./8)*aprime*m*e*cos(s-2*h+p) + aprime*m*m*cos(2*(s-h)))
    D = 1./((1./c1) + aprime1*e1*cos(h-p1))
    gm = (mu*M*r/(d*d*d))*(3*cos_theta**2-1) + (3./2)*(mu*M*r*r/(d*d*d*d))*(5*cos_theta**3 - 3*cos_theta)
    gs = mu*S*r/(D*D*D) * (3*cos_phi**2-1)

    love = (1+h2-1.5*k2)
    g0 = (gm+gs)*1e3*love
    return gm,gs,g0,g0

lat = 40.7914
lon = 282.1414
alt = 370*100.

def run_model(start_time,increment,days):
    times = []
    for i in range(int(24*7*3600/increment)):
        times.append(start_time + i*timedelta(seconds=increment))
    grav = []
    gmoon = []
    gsun = []
    dummy = []

    for time in times:
        gm,gs,g,d = solve_longman(lat,lon,alt,time)
        grav.append(g)
        gmoon.append(gm)
        gsun.append(gs)
        dummy.append(d)

    return gmoon,gsun,grav,dummy

t0 = datetime(2015,4,23,0,0,0)
net_results = np.loadtxt('net_results.txt',skiprows=10,usecols=[2])
net_time = np.arange(len(net_results))
plt.plot(net_time,net_results,color='r')

gm,gs,g,d = run_model(t0,600,7)
plt.plot(np.array(g),color='b')
plt.show()




# grav = []
# gmoon = []
# gsun = []
# dummy = []
#
# t0 = datetime(2015,4,23,0,0,0)
# times = []
# for i in range(7*24*6):
#     times.append(t0 + i*timedelta(seconds=600))
#
# for time in times:
#     gm,gs,g,d = solve_longman(lat,lon,alt,time)
#     grav.append(g)
#     gmoon.append(gm)
#     gsun.append(gs)
#     dummy.append(d)
#
# #plt.plot(grav)
# plt.plot(dummy)
