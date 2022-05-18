import model


test = model.OpinionFormation(N = 50, T = 3, nu = 3, alpha0 = 0, alpha1 = 1.2, deltax = 0.001, deltat = 0.01, bright = 0, bleft = 0) 
_,prob_2 = test.CrankNicolson()

# Generate Pseudo Time Series



