import longmantide as tide
from datetime import datetime

print "Run Single Time Point Example"
lat = 40.7914  # Station Latitude
lon = 282.1414  # Station Longitude
alt = 370.  # Station Altitude [meters]
model = tide.TideModel()  # Make a model object
time = datetime(2015, 4, 23, 0, 0, 0)  # When we want the tide
gm, gs, g = model.solve_longman(lat, lon, alt, time)
print gm, gs, g  # Lunar, Solar, and Total

print "\n\nRun Full 7 Day model"
model.increment = 60*10  # Run every 10 minutes
model.latitude = 40.7914  # Station Latitude
model.longitude = 282.1414  # Station Longitude
model.altitude = 370.  # Station Altitude [meters]
model.start_time = datetime(2015, 4, 23, 0, 0, 0)
model.duration = 7  # Model run duration [days]
model.run_model()  # Do the run
model.write('output.txt')  # Save results to text file
model.plot()  # Make a quick-dirty-plot


net_results = np.loadtxt('net_results.txt', skiprows=10, usecols=[2])
net_time = np.arange(len(net_results))
plt.plot(net_time, net_results, color='r')

gm, gs, g, d = run_model(t0, 600, 7)
plt.plot_date(model.model_times, model.gravity, '-k', linewidth=2)
plt.show()
