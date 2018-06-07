import numpy as np
from numba import jit

@jit
def cema_neige(Temp, Prec, params):
    
    '''
    Cema-Neige snow model
    Input:
    1. Data - timeseries:
        'T'- mean daily temperature (Celsium degrees)
        'P'- mean daily precipitation (mm/day)
    2. Params - list of model parameters:
        'CTG' - dimensionless weighting coefficient of the snow pack thermal state
                [0, 1]
        'Kf'  - day-degree rate of melting (mm/(day*celsium degree))
                [0.01, 10]
    Output:
    Total amount of liquid and melting precipitation daily timeseries
    (for coupling with hydrological model)
    '''

    FraqSolidPrecip = np.where(Temp < -0.2, 1, 0)

    CTG, Kf = params[0], params[1]

    ### initialization ###
    ## constants ##
    # melting temperature
    Tmelt = 0
    # Threshold for solid precip
    # function for Mean Annual Solid Precipitation
    #def MeanAnnualSolidPrecip(data):
    #annual_vals = [data[str(i)].loc[df["T"] < -0.2,  "P"].sum()\
    #               for i in np.unique(data.index.year)]
    #return np.mean(annual_vals)

    #MASP = MeanAnnualSolidPrecip(data)
    MASP = np.mean(Prec*FraqSolidPrecip)*365.25
    Gthreshold = 0.9*MASP
    MinSpeed = 0.1

    ## model states ##
    #G = 0
    #eTG = 0
    #PliqAndMelt = 0

    ### ouput of snow model
    # snowpack volume
    G = np.zeros(len(Temp))
    # snowpack termal state
    eTG = np.zeros(len(Temp))
    # Liquid precipitation
    Pliq = np.zeros(len(Temp))
    # Solid precipitation
    Psol = np.zeros(len(Temp))
    # Melt water
    Melt = np.zeros(len(Temp))
    # rain + snow water volume
    PliqAndMelt = np.zeros(len(Temp))

    for t in range(len(Temp)):
        ### solid and liquid precipitation accounting
        # liquid precipitation
        Pliq[t] = (1 - FraqSolidPrecip[t]) * Prec[t]
        # solid precipitation
        Psol[t] = FraqSolidPrecip[t] * Prec[t]
        ### Snow pack volume before melt
        G[t] = G[t-1] + Psol[t]
        ### Snow pack thermal state before melt
        eTG[t] = CTG * eTG[t-1] + (1 - CTG) * Temp[t]
        # control eTG
        if eTG[t] > 0: eTG[t] = 0
        ### potential melt
        if (int(eTG[t]) == 0) & (Temp[t] > Tmelt):
            PotMelt = Kf * (Temp[t] - Tmelt)
            if PotMelt > G[t]: PotMelt = G[t]
        else:
            PotMelt = 0
        ### ratio of snow pack cover (Gratio)
        if G[t] < Gthreshold:
            Gratio = G[t]/Gthreshold
        else:
            Gratio = 1
        ### actual melt
        Melt[t] = ((1 - MinSpeed) * Gratio + MinSpeed) * PotMelt
        ### snow pack volume update
        G[t] = G[t] - Melt[t]
        ### Gratio update
        if G[t] < Gthreshold:
            Gratio = G[t]/Gthreshold
        else:
            Gratio = 1

        ### Water volume to pass to the hydrological model
        PliqAndMelt[t] = Pliq[t] + Melt[t]

    return PliqAndMelt, G, eTG, Pliq, Psol, Melt

@jit
def _SS1(I,C,D):
    '''
    Values of the S curve (cumulative HU curve) of GR unit hydrograph UH1
    Inputs:
       C: time constant
       D: exponent
       I: time-step
    Outputs:
       SS1: Values of the S curve for I
    '''
    FI = I+1
    if FI <= 0: SS1 = 0
    elif FI < C: SS1 = (FI/C)**D
    else: SS1 = 1
    return SS1

@jit
def _SS2(I,C,D):
    '''
    Values of the S curve (cumulative HU curve) of GR unit hydrograph UH2
    Inputs:
       C: time constant
       D: exponent
       I: time-step
    Outputs:
       SS2: Values of the S curve for I
    '''
    FI = I+1
    if FI <= 0: SS2 = 0
    elif FI <= C: SS2 = 0.5*(FI/C)**D
    elif C < FI <= 2*C: SS2 = 1 - 0.5*(2 - FI/C)**D
    else: SS2 = 1
    return SS2

@jit
def _UH1(C, D, NH):
    '''
    C Computation of ordinates of GR unit hydrograph UH1 using successive differences on the S curve SS1
    C Inputs:
    C    C: time constant
    C    D: exponent
    C Outputs:
    C    OrdUH1: NH ordinates of discrete hydrograph
    '''
    OrdUH1 = np.zeros(NH)
    for i in range(NH):
        OrdUH1[i] = _SS1(i, C, D)-_SS1(i-1, C, D)
    return OrdUH1

@jit
def _UH2(C, D, NH):
    '''
    C Computation of ordinates of GR unit hydrograph UH1 using successive differences on the S curve SS1
    C Inputs:
    C    C: time constant
    C    D: exponent
    C Outputs:
    C    OrdUH1: NH ordinates of discrete hydrograph
    '''
    OrdUH2 = np.zeros(2*NH)
    for i in range(2*NH):
        OrdUH2[i] = _SS2(i, C, D)-_SS2(i-1, C, D)
    return OrdUH2

@jit
def gr4j(Temp, Prec, Evap, params):

    """
    Input:
    1. Meteorological forcing
        'T'- mean daily temperature (Celsium degrees)
        'P'- mean daily precipitation (mm/day)    
        'PET' - mean daily potential evaporation (mm/day)
    2. list of model parameters:
        GR4J params:
        X1 : production store capacity (X1 - PROD) [mm]
            [0.01, 3000]
        X2 : intercatchment exchange coefficient (X2 - CES) [mm/day]
            [-10, 10]
        X3 : routing store capacity (X3 - ROUT) [mm]
            [0.01, 1000]
        X4 : time constant of unit hydrograph (X4 - TB) [day]
            [0.01, 20.0]
        Cema-Neige snow model parameters:
        CTG : dimensionless weighting coefficient of the snow pack thermal state
            [0, 1]
        Kf : day-degree rate of melting (mm/(day*celsium degree))
            [0.01, 10]
    """
    # 1. parameters initialization
    X1, X2, X3, X4, CTG, Kf = params
    # 2. read the data
    Prec, G, eTG, Plig, Psol, Melt = cema_neige(Temp, Prec, [CTG, Kf])

    # parameter for unit hydrograph lenght
    NH = 20
    # 3. initialization of model states to zero
    # states of production St[0] and routing St[1] reservoirs holder
    # runoff
    Q = np.zeros(len(Prec))
    # production store capacity
    ProdStore = np.zeros(len(Prec))
    # routing store capacity
    RoutStore = np.zeros(len(Prec))
    # net rainfall
    PN = np.zeros(len(Prec))
    # actual evapotranspiration
    AE = np.zeros(len(Prec))
    # total intercatchment exchange
    AEXCH = np.zeros(len(Prec))
    
    #St = np.array([X1/2, X3/2])
    #ProdStore[0] = X1/2
    #RoutStore[0] = X3/2
    ProdStore[-1] = X1/2
    RoutStore[-1] = X3/2

    # Unit hydrograph states holders
    StUH1 = np.zeros(NH)
    StUH2 = np.zeros(2*NH)

    # 4. computation of UH ordinates
    OrdUH1 = _UH1(X4, 2.5, NH)
    OrdUH2 = _UH2(X4, 2.5, NH)

    # timestep implementation
    for t in range(len(Prec)):
        # interception and production store
        # check how connects Evap and Precip
        # case 1. No effective rainfall
        if Prec[t] <= Evap[t]:
            # net evapotranspiration capacity
            EN = Evap[t] - Prec[t]
            # net rainfall
            PN[t] = 0
            # part of production store capacity that suffers deficit
            WS = EN/X1
            # control WS
            if WS > 13: WS = 13
            TWS = np.tanh(WS)
            # part of production store capacity has an accumulated rainfall
            #Sr = St[0]/X1
            Sr = ProdStore[t-1]/X1
            # actual evaporation rate (will evaporate from production store)
            #ER = St[0]*(2 - Sr)*TWS/(1 + (1 - Sr)*TWS)
            ER = ProdStore[t-1]*(2 - Sr)*TWS/(1 + (1 - Sr)*TWS)
            # actual evapotranspiration
            AE[t] = ER + Prec[t]
            # production store capacity update
            #St[0] = St[0] - ER
            ProdStore[t] = ProdStore[t-1] - ER
            # control state of production store
            if ProdStore[t] < 0: ProdStore[t] = 0
            # water that reaches routing functions
            PR = 0
        # case 2. Effective rainfall produces runoff
        else:
            # net evapotranspiration capacity
            EN = 0
            # actual evapotranspiration
            AE[t] = Evap[t]
            # net rainfall
            PN[t] = Prec[t] - Evap[t]
            # part of production store capacity that holds rainfall
            WS = PN[t]/X1
            # control WS
            if WS > 13: WS = 13
            TWS = np.tanh(WS)
            # active part of production store
            Sr = ProdStore[t-1]/X1
            # amount of net rainfall that goes directly to the production store
            PS = X1*(1 - Sr*Sr)*TWS/(1 + Sr*TWS)
            # water that reaches routing functions
            PR = PN[t] - PS
            # production store capacity update
            ProdStore[t] = ProdStore[t-1] + PS
            # control state of production store
            if ProdStore[t] < 0: ProdStore[t] = 0

        # percolation from production store
        Sr = ProdStore[t]/X1
        Sr = Sr * Sr
        Sr = Sr * Sr
        # percolation leakage from production store
        PERC = ProdStore[t]*(1 - 1/np.sqrt(np.sqrt(1 + Sr/25.62891)))
        # production store capacity update
        ProdStore[t] = ProdStore[t] - PERC
        # update amount of water that reaches routing functions
        PR = PR + PERC

        # split of effective rainfall into the two routing components
        PRHU1 = PR*0.9
        PRHU2 = PR*0.1

        # convolution of unit hydrograph UH1
        for k in range(int( max(1, min(NH-1, int(X4+1))) )):
            StUH1[k] = StUH1[k+1] + OrdUH1[k] * PRHU1
        StUH1[NH-1] = OrdUH1[NH-1] * PRHU1

        # convolution of unit hydrograph UH2
        for k in range(int( max(1, min(2*NH-1, 2*int(X4+1))) )):
            StUH2[k] = StUH2[k+1] + OrdUH2[k] * PRHU2
        StUH2[2*NH-1] = OrdUH2[2*NH-1] * PRHU2

        # potential intercatchment semi-exchange
        # part of routing store
        #Rr = St[1]/X3
        Rr = RoutStore[t-1]/X3
        EXCH = X2*Rr*Rr*Rr*np.sqrt(Rr)

        # routing store
        AEXCH1 = EXCH
        if RoutStore[t-1] + StUH1[0] + EXCH < 0: 
            AEXCH1 = -RoutStore[t-1] - StUH1[0]
        # update state 2
        RoutStore[t] = RoutStore[t-1] + StUH1[0] + EXCH
        # control state 2
        if RoutStore[t] < 0: RoutStore[t] = 0
        Rr = RoutStore[t]/X3
        Rr = Rr * Rr
        Rr = Rr * Rr
        QR = RoutStore[t] * (1 - 1/np.sqrt(np.sqrt(1+Rr)))
        # update state 2
        RoutStore[t] = RoutStore[t] - QR

        # runoff from direct branch QD
        AEXCH2 = EXCH
        if StUH2[0] + EXCH < 0: AEXCH2 = -StUH2[0]
        QD = max(0, StUH2[0] + EXCH)
        
        # total intercatchment exchange
        AEXCH[t] = AEXCH1 + AEXCH2
        
        # total runoff
        Q[t] = QR + QD

    # control total runoff
    Q = np.where(Q != np.nan , Q, 0)
    Q = np.where(Q > 0, Q, 0)

    return Q

def gr4j_bounds():
    '''
    GR4J params:
    X1 : production store capacity (X1 - PROD) [mm]
        [0.01, 3000]
    X2 : intercatchment exchange coefficient (X2 - CES) [mm/day]
        [-10, 10]
    X3 : routing store capacity (X3 - ROUT) [mm]
        [0.01, 1000]
    X4 : time constant of unit hydrograph (X4 - TB) [day]
        [0.01, 10.0]
    Cema-Neige snow model parameters:
    CTG : dimensionless weighting coefficient of the snow pack thermal state
        [0, 1]
    Kf : day-degree rate of melting (mm/(day*celsium degree))
        [0.01, 10]
    '''
    bnds = ((0.01, 3000), (-10, 10), (0.01, 1000), (0.01, 30.0), (0, 1), (0.01, 10))
    return bnds