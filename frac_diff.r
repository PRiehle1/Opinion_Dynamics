############### A File for the estimation of the parameter of fractional differentiation #######
#  THE ESTIMATION AND APPLICATION OF LONG MEMORY TIME SERIES MODELS                            #
#   John Geweke, Susan Porter-Hudak (1983)                                                     #
################################################################################################
library(fracdiff)
library(tsqn)
library(robustbase)
library(MASS)
library(readxl)
library(LongMemoryTS)
library(dplyr)   

sim_Data <- read.csv(file = 'Validation_and_Statistics/Model_Simulations/Model_0/sim_0_set3.csv')
delta <- 0.5
T = 364
m <- floor(1+T^delta) 
iterations = 999
test <-data.matrix(sim_Data, rownames.force = NA)
output <- matrix(ncol=1, nrow=iterations)
  for(i in 1:iterations){
   output[i,] <- gph(test[i,], m = m)
}
write.matrix(output,file="d_sim_3_set_1.csv")
mean <- mean(output)

sim_Data <- read.csv(file = 'Validation_and_Statistics/Model_Simulations/Model_1/sim_1_set3.csv')
delta <- 0.5
T = 364
m <- floor(1+T^delta) 
iterations = 999
test <-data.matrix(sim_Data, rownames.force = NA)
output <- matrix(ncol=1, nrow=iterations)
  for(i in 1:iterations){
   output[i,] <- gph(test[i,], m = m)
}
write.matrix(output,file="d_sim_1_set_3.csv")
mean <- mean(output)

sim_Data <- read.csv(file = 'Validation_and_Statistics/Model_Simulations/Model_2/sim_2_set3.csv')
delta <- 0.5
T = 364
m <- floor(1+T^delta) 
iterations = 999
test <-data.matrix(sim_Data, rownames.force = NA)
output <- matrix(ncol=1, nrow=iterations)
  for(i in 1:iterations){
   output[i,] <- gph(test[i,], m = m)
}
write.matrix(output,file="d_sim_2_set_3.csv")
mean <- mean(output)

sim_Data <- read.csv(file = 'Validation_and_Statistics/Model_Simulations/Model_3/sim_3_set3.csv')
delta <- 0.5
T = 364
m <- floor(1+T^delta) 
iterations = 999
test <-data.matrix(sim_Data, rownames.force = NA)
output <- matrix(ncol=1, nrow=iterations)
  for(i in 1:iterations){
   output[i,] <- gph(test[i,], m = m)
}
write.matrix(output,file="d_sim_3_set_3.csv")
mean <- mean(output)

sim_Data <- read.csv(file = 'Validation_and_Statistics/Model_Simulations/Model_4/sim_4_set3.csv')
delta <- 0.5
T = 364
m <- floor(1+T^delta) 
iterations = 999
test <-data.matrix(sim_Data, rownames.force = NA)
output <- matrix(ncol=1, nrow=iterations)
  for(i in 1:iterations){
   output[i,] <- gph(test[i,], m = m)
}
write.matrix(output,file="d_sim_4_set_3.csv")
mean <- mean(output)

sim_Data <- read.csv(file = 'Validation_and_Statistics/Model_Simulations/Model_5/sim_5_set3.csv')
delta <- 0.5
T = 364
m <- floor(1+T^delta) 
iterations = 999
test <-data.matrix(sim_Data, rownames.force = NA)
output <- matrix(ncol=1, nrow=iterations)
  for(i in 1:iterations){
   output[i,] <- gph(test[i,], m = m)
}
write.matrix(output,file="d_sim_5_set_3.csv")
mean <- mean(output)

sim_Data <- read.csv(file = 'Validation_and_Statistics/Model_Simulations/Model_6/sim_6_set3.csv')
delta <- 0.5
T = 364
m <- floor(1+T^delta) 
iterations = 999
test <-data.matrix(sim_Data, rownames.force = NA)
output <- matrix(ncol=1, nrow=iterations)
  for(i in 1:iterations){
   output[i,] <- gph(test[i,], m = m)
}
write.matrix(output,file="d_sim_6_set_3.csv")
mean <- mean(output)

zew <- read.csv(file = 'Validation_and_Statistics/Model_Simulations/Model_6/sim_6_set3.csv')
output <- gph(test[i,], m = m)
write.matrix(output,file="d_zew.csv")