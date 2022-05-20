import model
import plot 


test = model.OpinionFormation(N = 50, T = 100, nu = 3, alpha0 = 0, alpha1 = 1.2, deltax = 0.001, deltat = 0.01, bright = 0, bleft = 0) 
prob,prob_end = test.CrankNicolson(x_0 = (-0.9))

plot_1 = plot.Plotting(param = prob, x = test.x, t = test.t)
plot_1.surface_plot()

# Generate Pseudo Time Series



