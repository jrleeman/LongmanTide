from longmantide import longmantide
from datetime import datetime

print "Run Single Time Point Example"
lat = 40.7914  # Station Latitude
lon = 282.1414  # Station Longitude
alt = 370.  # Station Altitude [meters]
model = longmantide.TideModel()  # Make a model object
time = datetime(2015, 4, 23, 0, 0, 0)  # When we want the tide
gm, gs, g = model.solve_longman(lat, lon, alt, time)
print gm, gs, g  # Lunar, Solar, and Total
