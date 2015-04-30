from  math import radians

def convertdms(d,m,s):
    d = float(d)
    m = float(m)
    s = float(s)
    return radians(d + m/60. + s/3600.)

print ""
print "Formula 10"
print "---------------------"
print "270 26\' 14.72\'\'            -> ", convertdms(270,26,14.72)
print "1336 rev +  1,108,411.20\'\' -> ", convertdms(1336*360.,0,1108411.20)
print "0 0\' 9.09\'\'                -> ", convertdms(0,0,9.09)
print "0 0\' 0.0068\'\'              -> ", convertdms(0,0,0.0068)

print ""
print "Formula 10\'"
print "---------------------"
print "270 26\' 11.72\'\'            -> ", convertdms(270,26,11.72)
print "1336 rev +  1,108,406.05\'\' -> ", convertdms(1336*360.,0,1108406.05)
print "0 0\' 7.128\'\'               -> ", convertdms(0,0,7.128)
print "0 0\' 0.0072\'\'              -> ", convertdms(0,0,0.0072)

print ""
print "Formula 11"
print "---------------------"
print "334 19\' 40.87\'\'       -> ", convertdms(334,19,40.87)
print "11 rev +  392515.94\'\' -> ", convertdms(11*360.,0,392515.94)
print "0 0\' 37.24\'\'          -> ", convertdms(0,0,37.24)
print "0 0\' 0.045\'\'          -> ", convertdms(0,0,0.045)

print ""
print "Formula 11\'"
print "---------------------"
print "334 19\' 46.42\'\'       -> ", convertdms(334,19,46.42)
print "11 rev +  392522.51\'\' -> ", convertdms(11*360.,0,392522.51)
print "0 0\' 37.15\'\'          -> ", convertdms(0,0,37.15)
print "0 0\' 0.036\'\'          -> ", convertdms(0,0,0.036)

print ""
print "Formula 12"
print "---------------------"
print "279 41\' 48.04\'\'  -> ", convertdms(279,41,48.04)
print "129,602,768.13\'\' -> ", convertdms(0,0,129602768.13)
print "0 0\' 1.089\'\'     -> ", convertdms(0,0,1.089)

print ""
print "Formula 12\'"
print "---------------------"
print "279 41\' 48.05\'\'  -> ", convertdms(279,41,48.05)
print "129,602,768.11\'\' -> ", convertdms(0,0,129602768.11)
print "0 0\' 1.080\'\'     -> ", convertdms(0,0,1.080)

print ""
print "Formula 19"
print "---------------------"
print "259 10\' 57.12\'\'       -> ", convertdms(259,10,57.12)
print "5 rev +  482,912.63\'\' -> ", convertdms(360*5.,0,482912.63)
print "0 0\' 7.58\'\'           -> ", convertdms(0,0,7.58)
print "0 0\' 0.008\'\'          -> ", convertdms(0,0,0.008)

print ""
print "Formula 19\'"
print "---------------------"
print "259 10\' 59.81\'\'       -> ", convertdms(259,10,59.81)
print "5 rev +  482,911.24\'\' -> ", convertdms(360*5.,0,482911.24)
print "0 0\' 7.48\'\'           -> ", convertdms(0,0,7.48)
print "0 0\' 0.007\'\'          -> ", convertdms(0,0,0.007)

print ""
print "Formula 26"
print "---------------------"
print "281 13\' 15.0\'\'   -> ", convertdms(281,13,15.0)
print "6,189.03\'\'       -> ", convertdms(0,0,6189.03)
print "0 0\' 1.63\'\'      -> ", convertdms(0,0,1.63)
print "0 0\' 0.012\'\'     -> ", convertdms(0,0,0.012)

print ""
print "Formula 26\'"
print "---------------------"
print "281 13\' 14.99\'\'  -> ", convertdms(281,13,14.99)
print "6,188.47\'\'       -> ", convertdms(0,0,6188.47)
print "0 0\' 1.62\'\'      -> ", convertdms(0,0,1.62)
print "0 0\' 0.011\'\'     -> ", convertdms(0,0,0.011)
