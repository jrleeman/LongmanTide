# Longman Tide

This is a module and tool that implements the tidal scheme of Longman 1959.

## Useage

An example use case of computing the tide over a 7 day period for State College, PA.

```
import longmantide as tide
from datetime import datetime

model = tide.TideModel()
model.start_time = datetime(2015,4,23,0,0,0)
model.increment = 60*10 # Run every 10 minutes
model.latitude = 40.7914
model.longitude = 282.1414
model.altitude = 370.
model.duration = 7
model.run_model()
model.write('output.txt')
model.plot()
```

Should you want to compute gravity at a specific time (say for correction of field data), that can be done as:

```
import longmantide as tide
from datetime import datetime

lat = 40.7914
lon = 282.1414
alt = 370.
model = tide.TideModel()
time = datetime(2015,4,23,0,0,0)
gm,gs,g = model.solve_longman(lat,lon,alt,time)
print gm,gs,g
```
