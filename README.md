# Longman Tide

This is a module and tool that implements the tidal scheme of Longman 1959.

## Useage

An example use case of computing the tide over a 7 day period for State College, PA.

```
import longmantide as tide
from datetime import datetime

model = tide.TideModel() # Make a model object
model.increment = 60*10 # Run every 10 minutes [seconds]
model.latitude = 40.7914 # Station Latitude
model.longitude = 282.1414 # Station Longitude
model.altitude = 370. # Station Altitude [meters]
model.start_time = datetime(2015,4,23,0,0,0)
model.duration = 7 # Model run duration [days]
model.run_model() # Do the run
model.write('output.txt') # Save results to text file
model.plot() # Make a quick-dirty-plot
```

Should you want to compute gravity at a specific time (say for correction of field data), that can be done as:

```
import longmantide as tide
from datetime import datetime

lat = 40.7914 # Station Latitude
lon = 282.1414 # Station Longitude
alt = 370. # Station Altitude [meters]
model = tide.TideModel() # Make a model object
time = datetime(2015,4,23,0,0,0) # When we want the tide
gm,gs,g = model.solve_longman(lat,lon,alt,time)
print gm,gs,g # Lunar, Solar, and Total
```
