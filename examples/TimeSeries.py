from longmantide import longmantide
from datetime import datetime

print "\n\nRun Full 7 Day model"
model = longmantide.TideModel()  # Make a model object
model.increment = 60*10  # Run every 10 minutes
model.latitude = 40.7914  # Station Latitude
model.longitude = 282.1414  # Station Longitude
model.altitude = 370.  # Station Altitude [meters]
model.start_time = datetime(2015, 4, 23, 0, 0, 0)
model.duration = 7  # Model run duration [days]
model.run_model()  # Do the run
model.write('output.txt')  # Save results to text file
model.plot()  # Make a quick-dirty-plot
