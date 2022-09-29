"""
This file determines the order of accuary of the Crank-Nicolson scheme used to approximate the FPE. We use the method probosed by Osterby(1998)
"""
### Import Packages ####
from model import OpinionFormation
import numpy as np

def run():
    """
    It runs the Crank-Nicolson method for different values of T and dx, and then calculates the order of
    accuracy.
    """
    ############################################################################################################################
    #                                       Time Direction
    ############################################################################################################################
    #x(0) = 0
    ### Check order of accuracy in time without dirft ####
    quotient = []
    for t in (0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25):
        test_1 = OpinionFormation(N = 50, T =t, nu = 3 , alpha0 = 0 , alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.0025, deltat = 1/100, model_type =0)    #
        test_2 =OpinionFormation(N = 50, T =t, nu = 3 , alpha0 = 0, alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.0025, deltat = 1/200, model_type =0) 
        test_3 = OpinionFormation(N = 50, T = t , nu = 3 , alpha0 = 0 , alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.0025, deltat = 1/400, model_type =0)    #


        area_1, prob_1,prob_end_1 = test_1.CrankNicolson(x_0 = 0,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)
        area_2, prob_2,prob_end_2 = test_2.CrankNicolson(x_0 = 0,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)
        area_3, prob_3,prob_end_3 = test_3.CrankNicolson(x_0 = 0,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)

        quotient.append(np.round(np.divide(np.subtract(prob_end_2,prob_end_1),np.subtract(prob_end_3,prob_end_2)),1))
    q = np.asarray(quotient)
    final_q = q[:,(0,100,200,300,400,500,600,700,800)]
    np.savetxt(r"Validation_and_Statistics/Crank_Nicolson/order_acc_k_set_1.csv", final_q, delimiter=",",fmt ='% s')
    print("The order of accuracy rows:T, columns (-1,1)\n" + str(final_q))
    ### Check order of accuracy in time with dirft ###
    quotient = []
    for t in (0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25):
        test_1 = OpinionFormation(N = 50, T =t, nu = 3 , alpha0 = 0.02 , alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.0025, deltat = 1/100, model_type =0)    #
        test_2 =OpinionFormation(N = 50, T =t, nu = 3 , alpha0 = 0.02, alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.0025, deltat = 1/200, model_type =0) 
        test_3 = OpinionFormation(N = 50, T = t , nu = 3 , alpha0 = 0.02 , alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.0025, deltat = 1/400, model_type =0)    #


        area_1, prob_1,prob_end_1 = test_1.CrankNicolson(x_0 = 0,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)
        area_2, prob_2,prob_end_2 = test_2.CrankNicolson(x_0 = 0,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)
        area_3, prob_3,prob_end_3 = test_3.CrankNicolson(x_0 = 0,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)

        quotient.append(np.round(np.divide(np.subtract(prob_end_2,prob_end_1),np.subtract(prob_end_3,prob_end_2)),1))
    q = np.asarray(quotient)
    final_q = q[:,(0,100,200,300,400,500,600,700,800)]
    np.savetxt(r"Validation_and_Statistics/Crank_Nicolson/order_acc_k_set_2.csv", final_q, delimiter=",",fmt ='% s')
    print("The order of accuracy rows:T, columns (-1,1)\n" + str(final_q))
    ############################################################################################################################
    #
    # x(0) = 0.8
    #
    #### Check order of accuracy in time without dirft ####
    quotient = []
    for t in (0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25):
        test_1 = OpinionFormation(N = 50, T =t, nu = 3 , alpha0 = 0 , alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.0025, deltat = 1/100, model_type =0)    #
        test_2 =OpinionFormation(N = 50, T =t, nu = 3 , alpha0 = 0, alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.0025, deltat = 1/200, model_type =0) 
        test_3 = OpinionFormation(N = 50, T = t , nu = 3 , alpha0 = 0 , alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.0025, deltat = 1/400, model_type =0)    #


        area_1, prob_1,prob_end_1 = test_1.CrankNicolson(x_0 = -0.8,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)
        area_2, prob_2,prob_end_2 = test_2.CrankNicolson(x_0 = -0.8,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)
        area_3, prob_3,prob_end_3 = test_3.CrankNicolson(x_0 = -0.8,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)

        quotient.append(np.round(np.divide(np.subtract(prob_end_2,prob_end_1),np.subtract(prob_end_3,prob_end_2)),1))
    q = np.asarray(quotient)
    final_q = q[:,(0,100,200,300,400,500,600,700,800)]
    np.savetxt(r"Validation_and_Statistics/Crank_Nicolson/order_acc_k_set_3.csv", final_q, delimiter=",",fmt ='% s')
    print("The order of accuracy rows:T, columns (-1,1)\n" + str(final_q))
    ## Check order of accuracy in time with dirft ###
    quotient = []
    for t in (0.25,0.5,0.75,1,1.25,1.5,1.75,2,2.25):
        test_1 = OpinionFormation(N = 50, T =t, nu = 3 , alpha0 = 0.02 , alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.0025, deltat = 1/100, model_type =0)    #
        test_2 =OpinionFormation(N = 50, T =t, nu = 3 , alpha0 = 0.02, alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.0025, deltat = 1/200, model_type =0) 
        test_3 = OpinionFormation(N = 50, T = t , nu = 3 , alpha0 = 0.02 , alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.0025, deltat = 1/400, model_type =0)    #


        area_1, prob_1,prob_end_1 = test_1.CrankNicolson(x_0 = -0.8,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)
        area_2, prob_2,prob_end_2 = test_2.CrankNicolson(x_0 = -0.8,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)
        area_3, prob_3,prob_end_3 = test_3.CrankNicolson(x_0 = -0.8,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)

        quotient.append(np.round(np.divide(np.subtract(prob_end_2,prob_end_1),np.subtract(prob_end_3,prob_end_2)),1))
    q = np.asarray(quotient)
    final_q = q[:,(0,100,200,300,400,500,600,700,800)]
    np.savetxt(r"Validation_and_Statistics/Crank_Nicolson/order_acc_k_set_4.csv", final_q, delimiter=",",fmt ='% s')
    print("The order of accuracy rows:T, columns (-1,1)\n" + str(final_q))
    #############################################################################################################################################################
    #                   Space Direction                                                                                                                         #
    ##############################################################################################################################################################
    quotient = []
    for t in (0.25,0.5,0.75,1,1.25,1.5,1.75,2):
        test_1 = OpinionFormation(N = 50, T =t, nu = 3 , alpha0 = 0 , alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.0025, deltat = 1/100, model_type =0)    #
        test_2 =OpinionFormation(N = 50, T =t, nu = 3 , alpha0 = 0, alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.00125, deltat = 1/100, model_type =0) 
        test_3 = OpinionFormation(N = 50, T = t , nu = 3 , alpha0 = 0 , alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.000625, deltat = 1/100, model_type =0)    #


        area_1, prob_1,prob_end_1 = test_1.CrankNicolson(x_0 = 0,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)
        area_2, prob_2,prob_end_2 = test_2.CrankNicolson(x_0 = 0,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)
        area_3, prob_3,prob_end_3 = test_3.CrankNicolson(x_0 = 0,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)

        quotient.append(np.round(np.divide(np.subtract(prob_end_2[[0,int(0.25/test_2.dx),int(0.5/test_2.dx),int(0.75/test_2.dx),int(1/test_2.dx),int(1.25/test_2.dx),int(1.5/test_2.dx), int(1.75/test_2.dx),int(2/test_2.dx)]],
        prob_end_1[[0,int(0.25/test_1.dx),int(0.5/test_1.dx),int(0.75/test_1.dx),int(1/test_1.dx),int(1.25/test_1.dx),int(1.5/test_1.dx),int(1.75/test_1.dx),int(2/test_1.dx)]]),
        np.subtract(prob_end_3[[0,int(0.25/test_3.dx),int(0.5/test_3.dx),int(0.75/test_3.dx),int(1/test_3.dx),int(1.25/test_3.dx),int(1.5/test_3.dx),int(1.75/test_3.dx),int(2/test_3.dx)]],prob_end_2[[0,int(0.25/test_2.dx),int(0.5/test_2.dx),int(0.75/test_2.dx),int(1/test_2.dx),int(1.25/test_2.dx),int(1.5/test_2.dx), int(1.75/test_2.dx),int(2/test_2.dx)]])),1))
    q = np.asarray(quotient)
    np.savetxt(r"Validation_and_Statistics/Crank_Nicolson/order_acc_h_set_1.csv", q, delimiter=",",fmt ='% s')
    print("The order of accuracy rows:T, columns (-1,1)\n" + str(q))
    ### Check order of accuracy in time with dirft ###
    quotient = []
    for t in (0.25,0.5,0.75,1,1.25,1.5,1.75,2):
        test_1 = OpinionFormation(N = 50, T =t, nu = 3 , alpha0 = 0.02 , alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.0025, deltat = 1/100, model_type =0)    #
        test_2 =OpinionFormation(N = 50, T =t, nu = 3 , alpha0 = 0.02, alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.00125, deltat = 1/100, model_type =0) 
        test_3 = OpinionFormation(N = 50, T = t , nu = 3 , alpha0 = 0.02 , alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.000625, deltat = 1/100, model_type =0)    #


        area_1, prob_1,prob_end_1 = test_1.CrankNicolson(x_0 = 0,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)
        area_2, prob_2,prob_end_2 = test_2.CrankNicolson(x_0 = 0,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)
        area_3, prob_3,prob_end_3 = test_3.CrankNicolson(x_0 = 0,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)

        quotient.append(np.round(np.divide(np.subtract(prob_end_2[[0,int(0.25/test_2.dx),int(0.5/test_2.dx),int(0.75/test_2.dx),int(1/test_2.dx),int(1.25/test_2.dx),int(1.5/test_2.dx), int(1.75/test_2.dx),int(2/test_2.dx)]],
        prob_end_1[[0,int(0.25/test_1.dx),int(0.5/test_1.dx),int(0.75/test_1.dx),int(1/test_1.dx),int(1.25/test_1.dx),int(1.5/test_1.dx),int(1.75/test_1.dx),int(2/test_1.dx)]]),
        np.subtract(prob_end_3[[0,int(0.25/test_3.dx),int(0.5/test_3.dx),int(0.75/test_3.dx),int(1/test_3.dx),int(1.25/test_3.dx),int(1.5/test_3.dx),int(1.75/test_3.dx),int(2/test_3.dx)]],prob_end_2[[0,int(0.25/test_2.dx),int(0.5/test_2.dx),int(0.75/test_2.dx),int(1/test_2.dx),int(1.25/test_2.dx),int(1.5/test_2.dx), int(1.75/test_2.dx),int(2/test_2.dx)]])),1))
    q = np.asarray(quotient)
    np.savetxt(r"Validation_and_Statistics/Crank_Nicolson/order_acc_h_set_2.csv", q, delimiter=",",fmt ='% s')
    print("The order of accuracy rows:T, columns (-1,1)\n" + str(q))
    ############################################################################################################################
    #
    # x(0) = -0.8
    #
    #### Check order of accuracy in time without dirft ####
    quotient = []
    for t in (0.25,0.5,0.75,1,1.25,1.5,1.75,2):
        test_1 = OpinionFormation(N = 50, T =t, nu = 3 , alpha0 = 0 , alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.0025, deltat = 1/100, model_type =0)    #
        test_2 =OpinionFormation(N = 50, T =t, nu = 3 , alpha0 = 0, alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.00125, deltat = 1/100, model_type =0) 
        test_3 = OpinionFormation(N = 50, T = t , nu = 3 , alpha0 = 0 , alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.000625, deltat = 1/100, model_type =0)    #


        area_1, prob_1,prob_end_1 = test_1.CrankNicolson(x_0 = -0.8,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)
        area_2, prob_2,prob_end_2 = test_2.CrankNicolson(x_0 = -0.8,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)
        area_3, prob_3,prob_end_3 = test_3.CrankNicolson(x_0 = -0.8,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)

        quotient.append(np.round(np.divide(np.subtract(prob_end_2[[0,int(0.25/test_2.dx),int(0.5/test_2.dx),int(0.75/test_2.dx),int(1/test_2.dx),int(1.25/test_2.dx),int(1.5/test_2.dx), int(1.75/test_2.dx),int(2/test_2.dx)]],
        prob_end_1[[0,int(0.25/test_1.dx),int(0.5/test_1.dx),int(0.75/test_1.dx),int(1/test_1.dx),int(1.25/test_1.dx),int(1.5/test_1.dx),int(1.75/test_1.dx),int(2/test_1.dx)]]),
        np.subtract(prob_end_3[[0,int(0.25/test_3.dx),int(0.5/test_3.dx),int(0.75/test_3.dx),int(1/test_3.dx),int(1.25/test_3.dx),int(1.5/test_3.dx),int(1.75/test_3.dx),int(2/test_3.dx)]],prob_end_2[[0,int(0.25/test_2.dx),int(0.5/test_2.dx),int(0.75/test_2.dx),int(1/test_2.dx),int(1.25/test_2.dx),int(1.5/test_2.dx), int(1.75/test_2.dx),int(2/test_2.dx)]])),1))
    q = np.asarray(quotient)
    np.savetxt("Validation_and_Statistics/order_acc_h_set_3.csv", q, delimiter=",",fmt ='% s')
    print("The order of accuracy rows:T, columns (-1,1)\n" + str(q))
    ### Check order of accuracy in time with dirft ###
    quotient = []
    for t in (0.25,0.5,0.75,1,1.25,1.5,1.75,2):
        test_1 = OpinionFormation(N = 50, T =t, nu = 3 , alpha0 = 0.02 , alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.0025, deltat = 1/100, model_type =0)    #
        test_2 =OpinionFormation(N = 50, T =t, nu = 3 , alpha0 = 0.02, alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.00125, deltat = 1/100, model_type =0) 
        test_3 = OpinionFormation(N = 50, T = t , nu = 3 , alpha0 = 0.02 , alpha1 = 1.2 ,alpha2 = None ,alpha3 =  None, deltax = 0.000625, deltat = 1/100, model_type =0)    #


        area_1, prob_1,prob_end_1 = test_1.CrankNicolson(x_0 = -0.8,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)
        area_2, prob_2,prob_end_2 = test_2.CrankNicolson(x_0 = -0.8,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)
        area_3, prob_3,prob_end_3 = test_3.CrankNicolson(x_0 = -0.8,  y = None, x_l = None ,calc_dens = True, converged= False, fast_comp = False)

        quotient.append(np.round(np.divide(np.subtract(prob_end_2[[0,int(0.25/test_2.dx),int(0.5/test_2.dx),int(0.75/test_2.dx),int(1/test_2.dx),int(1.25/test_2.dx),int(1.5/test_2.dx), int(1.75/test_2.dx),int(2/test_2.dx)]],
        prob_end_1[[0,int(0.25/test_1.dx),int(0.5/test_1.dx),int(0.75/test_1.dx),int(1/test_1.dx),int(1.25/test_1.dx),int(1.5/test_1.dx),int(1.75/test_1.dx),int(2/test_1.dx)]]),
        np.subtract(prob_end_3[[0,int(0.25/test_3.dx),int(0.5/test_3.dx),int(0.75/test_3.dx),int(1/test_3.dx),int(1.25/test_3.dx),int(1.5/test_3.dx),int(1.75/test_3.dx),int(2/test_3.dx)]],prob_end_2[[0,int(0.25/test_2.dx),int(0.5/test_2.dx),int(0.75/test_2.dx),int(1/test_2.dx),int(1.25/test_2.dx),int(1.5/test_2.dx), int(1.75/test_2.dx),int(2/test_2.dx)]])),1))
    q = np.asarray(quotient)
    np.savetxt(r"Validation_and_Statistics/Crank_Nicolson/order_acc_h_set_4.csv", q, delimiter=",",fmt ='% s')
    print("The order of accuracy rows:T, columns (-1,1)\n" + str(q))