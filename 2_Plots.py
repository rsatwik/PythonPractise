%pylab # To intialize the pylab package
linspace? # help for linspace command
t=linspace(-pi,pi,100) # (start, end, number of elements) 100 numbers between -pi and pi
len(t) # returns 100
cosine=cos(t)
plot(t,cosine)
clf() # to clear plots and avoid overlapping of plots in the same figure
plot(t,sin(t)*sin(t)/t)