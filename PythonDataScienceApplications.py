# Manipulating lists
# Creates the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Corrects the bathroom area
areas[9]=10.5

# Changes "living room" to "chill zone"
areas[4]="chill zone"

# Creates lists first and second
first = [11.25, 18.0, 20.0]
second = [10.75, 9.50]

# Pastes together first and second
full = first + second

# Sorts full in descending order
full_sorted = sorted(full, reverse=True)

# Practice with Numpy
# updates as 2D Numpy array

# Imports numpy package
import numpy as np

# Creates np_baseball (3 cols)
np_baseball = np.array(baseball)

# Prints out addition of np_baseball and update
print(np_baseball + update)

# Creates Numpy array: conversion
conversion = np.array([.0254, .453592, 1])

# Prints out product of np_baseball and conversion
print(np_baseball * conversion)

#Using Numpy with Statistics

# Imports numpy
import numpy as np

# Converts positions and heights to numpy arrays: np_positions, np_heights
np_heights = np.array(heights)
np_positions = np.array(positions)


# Heights of the goalkeepers: gk_heights
gk_heights = np_heights[np_positions=='GK']

# Heights of the other players: other_heights
other_heights = np_heights[np_positions != 'GK']

# Prints out the median height of goalkeepers
print("Median height of goalkeepers: " + str(np.median(gk_heights)))

# Prints out the median height
print("Median height of other players: " + str(np.median(other_heights)))

#Plotting with Pyplot

# Imports matplotlib
Import matplotlib.pyplot as plt

# Makes a line plot
plt.plot(year, pop)

# Prints the last item of gdp_cap and life_exp
print(gdp_cap[-1])
print(life_exp[-1])

# Makes a line plot
plt.plot(gdp_cap,life_exp)

# Displays the plot
plt.show()

# Changes the line plot below to a scatter plot
plt.scatter(gdp_cap, life_exp)

# Puts the x-axis on a logarithmic scale
plt.xscale('log')

# Shows plot
plt.show()

# Practice with Histograms
# Builds histogram with 5 bins

plt.hist(life_exp, bins = 5)
# Shows and cleans up plot
plt.show()
plt.clf()

# Builds histogram with 20 bins
plt.hist(life_exp, bins = 20)

# Show and clean 
plt.show()
plt.clf()

#Practice with Customizing files
# Basic scatter plot, log scale
plt.scatter(gdp_cap, life_exp)
plt.xscale('log') 

# Strings
xlab = 'GDP per Capita [in USD]'
ylab = 'Life Expectancy [in years]'
title = 'World Development in 2007'

# Add axis labels
plt.xlabel(xlab)
plt.ylabel(ylab)

# Adds title
plt.title(title)

# Displays the plot
plt.show()

# Scatter plot
plt.scatter(gdp_cap, life_exp)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')

# Definition of tick_val and tick_lab
tick_val = [1000,10000,100000]
tick_lab = ['1k','10k','100k']

plt.xticks(tick_val, tick_lab)

# Displays the plot
plt.show()

# Scatter plot
plt.scatter(x = gdp_cap, y = life_exp, s = np.array(pop) * 2, c = col, alpha = 0.8)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000,10000,100000], ['1k','10k','100k'])

# Additional customizations
plt.text(1550, 71, 'India')
plt.text(5700, 80, 'China')

# Adds grid() call
plt.grid(True)

# Shows the plot
plt.show()

#Logic and Control Flow

room = "bed"
area = 14.0

# if-elif-else construct for room
if room == "kit" :
    print("looking around in the kitchen.")
elif room == "bed":
    print("looking around in the bedroom.")
else :
    print("looking around elsewhere.")

# if-elif-else construct for area
if area > 15 :
    print("big place!")
elif area > 10 : 
    print("medium size,nice!")
else :
    print("pretty small.")
	
#CVS to Dataframe
# Import pandas
import pandas as pd

# Imports the cars.csv data
cars = pd.read_csv('cars.csv')

# Display
print(cars)
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Prints out observation for Japan
print(cars.loc['JAP'])

# Prints out observations for Australia and Egypt
print(cars.loc[['AUS', 'EG']])

# Import cars
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Prints out drives_right value of Morocco
print(cars.loc['MOR', 'drives_right'])

# Prints sub-DataFrame
print(cars.loc[['RU', 'MOR'], ['country', 'drives_right']])