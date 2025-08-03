# -*- coding: utf-8 -*-
"""
Created on Thu Jun 06 2024

@author: R Mike L Evans
"""

import numpy as np
from numpy import pi
from numpy import sin
from numpy import cos
from scipy.optimize import minimize

#*************************
#User-defined parameters:
NPassiveCoords = 1
NActiveCoords = 1
t0 = 0.0
t1 = 2.0
#Initial positions:
q0 = [0.8]
alpha0 = [0.0]
#Initial velocities:
qDot0 = [-1.0]
alphaDot0 = [1.5]
#Final positions:
q1 = [0.27788]
alpha1 = [2.2386683154980482]
#Final velocities:
qDot1 = [1.4204653029035241]
alphaDot1 = [1.3124052038675633]
#*************************
#Other parameters:
monkey = False    # Selects between Lazy Monkey model or Toy Model 2
useFourierConstraints = False
maxIterations = 200
NParamsPerActiveCoord = 10
NParamsPerPassiveCoord = NParamsPerActiveCoord*10  # Must be larger than NParamsPerActiveCoord
NParams = NParamsPerPassiveCoord*NPassiveCoords+NParamsPerActiveCoord*NActiveCoords
NConstraintsPerPassiveCoord = NParamsPerPassiveCoord + 2
initialGuess = np.zeros(NParams)
w = pi/(t1-t0)
a0 = np.empty(NPassiveCoords+NActiveCoords, dtype=float)
a1 = np.empty(NPassiveCoords+NActiveCoords, dtype=float)
a2 = np.empty(NPassiveCoords+NActiveCoords, dtype=float)
a3 = np.empty(NPassiveCoords+NActiveCoords, dtype=float)
for i in range(NPassiveCoords):
    a0[i] = 0.5*(q0[i]+q1[i]) + ((qDot1[i]-qDot0[i])/w)
    a1[i] = 0.5*(q0[i]-q1[i]) + ((qDot0[i]+qDot1[i])/w)
    a2[i] = 2*qDot0[i]/w
    a3[i] = -2*qDot1[i]/w
for i in range(NActiveCoords):
    a0[i+NPassiveCoords] = 0.5*(alpha0[i]+alpha1[i]) + ((alphaDot1[i]-alphaDot0[i])/w)
    a1[i+NPassiveCoords] = 0.5*(alpha0[i]-alpha1[i]) + ((alphaDot0[i]+alphaDot1[i])/w)
    a2[i+NPassiveCoords] = 2*alphaDot0[i]/w
    a3[i+NPassiveCoords] = -2*alphaDot1[i]/w
#*************************

def unpack(Q,QDot,QDotDot):
    q = Q[:NPassiveCoords]
    a = Q[NPassiveCoords:]
    qDot = QDot[:NPassiveCoords]
    aDot = QDot[NPassiveCoords:]
    qDotDot = QDotDot[:NPassiveCoords]
    aDotDot = QDotDot[NPassiveCoords:]
    return q, a, qDot, aDot, qDotDot, aDotDot

#************************************************************
#*******User-defined functions for the specific system*******

def P(Q,QDot,QDotDot):  # The power function
    q, a, qDot, aDot, qDotDot, aDotDot = unpack(Q, QDot, QDotDot)
    # User updates the following line for the particular system
    if monkey:
        output = aDot[0]*(sin(q[0]+a[0])+3*qDot[0]*qDot[0]*sin(a[0])+qDotDot[0]*(2.0+3*cos(a[0]))+2*aDotDot[0])
    else:
        output = aDot[0]*(qDotDot[0]+aDotDot[0])
    return output

def P_Q(Q,QDot,QDotDot):   # Partial derivatives of P with respect to Q
    q, a, qDot, aDot, qDotDot, aDotDot = unpack(Q, QDot, QDotDot)
    output = np.empty(NPassiveCoords+NActiveCoords, dtype=float)
    # User updates the following lines for the particular system
    if monkey:
        output[0] = aDot[0]*cos(q[0]+a[0])
        output[1] = output[0]+aDot[0]*(+3*qDot[0]*qDot[0]*cos(a[0])-3*qDotDot[0]*sin(a[0]))
    else:
        output[0] = 0.0
        output[1] = 0.0
    return output

def P_QDot(Q,QDot,QDotDot):   # Partial derivatives of P with respect to QDot
    q, a, qDot, aDot, qDotDot, aDotDot = unpack(Q, QDot, QDotDot)
    output = np.empty(NPassiveCoords+NActiveCoords, dtype=float)
    # User updates the following lines for the particular system
    if monkey:
        output[0] = 6*qDot[0]*aDot[0]*sin(a[0])
        output[1] = sin(q[0]+a[0])+3*qDot[0]*qDot[0]*sin(a[0])+qDotDot[0]*(2.0+3*cos(a[0]))+2*aDotDot[0]
    else:
        output[0] = 0.0
        output[1] = qDotDot[0]+aDotDot[0]
    return output

def P_QDotDot(Q,QDot,QDotDot):   # Partial derivatives of P with respect to QDotDot
    q, a, qDot, aDot, qDotDot, aDotDot = unpack(Q, QDot, QDotDot)
    output = np.empty(NPassiveCoords+NActiveCoords, dtype=float)
    # User updates the following lines for the particular system
    if monkey:
        output[0] = aDot[0]*(2.0+3*cos(a[0]))
        output[1] = 2*aDot[0]
    else:
        output[0] = aDot[0]
        output[1] = aDot[0]
    return output

def A(component,Q,QDot,QDotDot):   # The Euler-Lagrange function for any passive Coord (the component number)
    q, a, qDot, aDot, qDotDot, aDotDot = unpack(Q, QDot, QDotDot)
    output = np.empty(NPassiveCoords, dtype=float)
    # User updates the following lines for the particular system
    if component == 0:
        if monkey:
            output = 3*sin(q[0])+sin(q[0]+a[0])+10*qDotDot[0]+2*aDotDot[0]+3*(2*qDotDot[0]+aDotDot[0])*cos(a[0])-3*aDot[0]*(2*qDot[0]+aDot[0])*sin(a[0])
        else:
            output = q[0]-a[0] + qDotDot[0]
    return output

def A_Q(component,Q,QDot,QDotDot):   # Partial derivatives of a component of A with respect to Q
    q, a, qDot, aDot, qDotDot, aDotDot = unpack(Q, QDot, QDotDot)
    output = np.empty(NPassiveCoords+NActiveCoords, dtype=float)
    # User updates the following lines for the particular system
    if component == 0:
        if monkey:
            output[0] = 3*cos(q[0])+cos(q[0]+a[0])
            output[1] = cos(q[0]+a[0])-3*(2*qDotDot[0]+aDotDot[0])*sin(a[0])-3*aDot[0]*(2*qDot[0]+aDot[0])*cos(a[0])
        else:
            output[0] = 1.0
            output[1] = -1.0
    return output

def A_QDot(component,Q,QDot,QDotDot):   # Partial derivatives of a component of A A with respect to QDot
    q, a, qDot, aDot, qDotDot, aDotDot = unpack(Q, QDot, QDotDot)
    output = np.empty(NPassiveCoords+NActiveCoords, dtype=float)
    # User updates the following lines for the particular system
    if component == 0:
        if monkey:
            output[0] = -6*aDot[0]*sin(a[0])
            output[1] = -6*(qDot[0]+aDot[0])*sin(a[0])
        else:
            output[0] = 0.0
            output[1] = 0.0
    return output

def A_QDotDot(component,Q,QDot,QDotDot):   # Partial derivatives of a component of A A with respect to QDotDot
    q, a, qDot, aDot, qDotDot, aDotDot = unpack(Q, QDot, QDotDot)
    output = np.empty(NPassiveCoords+NActiveCoords, dtype=float)
    # User updates the following lines for the particular system
    if component == 0:
        if monkey:
            output[0] = 10+6*cos(a[0])
            output[1] = 2+3*cos(a[0])
        else:
            output[0] = 1.0
            output[1] = 0.0
    return output

def f(x):  # The effort rate utility function
    return x*x

def fPrime(x):  # Derivative of f(x)
    return 2*x

#************************************************************

def SimpsonsRule(f):
    global t0,t1
    NIntervals = NParamsPerActiveCoord*2  # Must be even
    dt = (t1-t0)/NIntervals
    t = t0+dt
    temp = f(t)
    oddTotal = np.zeros(np.shape(temp))
    oddTotal = np.add(oddTotal,temp)
    evenTotal = np.zeros(np.shape(temp))
    for i in range(2, NIntervals, 2):
        t += dt
        evenTotal = np.add(evenTotal,f(t))
        t += dt
        oddTotal = np.add(oddTotal,f(t))
    output = np.add(np.multiply(4,oddTotal),np.multiply(2,evenTotal))
    output = np.add(output,f(t0))
    output = np.add(output,f(t1))
    output = np.multiply(output,dt/3.0)
    return output

def NModes(Coord):
    if Coord<NPassiveCoords:
        return NParamsPerPassiveCoord
    else:
        return NParamsPerActiveCoord
    
def NCoords(mode):
    if mode<NParamsPerActiveCoord:
        return NPassiveCoords+NActiveCoords
    else:
        return NPassiveCoords

def paramIndex(Coord,mode):
    if Coord<NPassiveCoords:
        return Coord*NParamsPerPassiveCoord + mode
    else:
        return NPassiveCoords*NParamsPerPassiveCoord + (Coord-NPassiveCoords)*NParamsPerActiveCoord + mode

def trajectory(t,paramList):
    Q = np.empty(NPassiveCoords+NActiveCoords, dtype=float)
    QDot = np.empty(NPassiveCoords+NActiveCoords, dtype=float)
    QDotDot = np.empty(NPassiveCoords+NActiveCoords, dtype=float)
    wt = w*(t-t0)
    sinwt = sin(wt)
    coswt = cos(wt)
    sinhalfwt = sin(wt/2)
    coshalfwt = cos(wt/2)
    for Coord in range(NPassiveCoords+NActiveCoords):
        sum0 = 0
        sum1 = 0
        sum2 = 0
        for mode in range(NModes(Coord)):
            k = mode+1
            bk = paramList[paramIndex(Coord,mode)]
            kwt = k*wt
            sinkwt = sin(kwt)
            coskwt = cos(kwt)
            sum0 += bk * sinkwt
            sum1 += bk * k*coskwt
            sum2 += bk * k*k*sinkwt
        Q[Coord] = a0[Coord] + a1[Coord]*coswt + a2[Coord]*sinhalfwt + a3[Coord]*coshalfwt + sinwt*sum0
        QDot[Coord] = -a1[Coord]*sinwt + a2[Coord]*coshalfwt/2 - a3[Coord]*sinhalfwt/2 + coswt*sum0 + sinwt*sum1
        QDot[Coord] *= w
        QDotDot[Coord] = -a1[Coord]*coswt - a2[Coord]*sinhalfwt/4 - a3[Coord]*coshalfwt/4 - sinwt*sum0 +2*coswt*sum1 - sinwt*sum2
        QDotDot[Coord] *= w*w
    return Q, QDot, QDotDot

def effort(paramList):
    return SimpsonsRule(lambda t,x=paramList: f(P(*trajectory(t,x))))

def FJIntegrand(Q,QDot,QDotDot,t):
    output = np.empty(NParams, dtype=float)
    PQ = P_Q(Q,QDot,QDotDot)
    PQD = P_QDot(Q,QDot,QDotDot)
    PQDD = P_QDotDot(Q,QDot,QDotDot)
    fp = fPrime(P(Q,QDot,QDotDot))
    wt = w*(t-t0)
    sinwt = sin(wt)
    coswt = cos(wt)
    for mode in range(NParamsPerPassiveCoord):
        k = mode+1
        sinkwt = sin(k*wt)
        coskwt = cos(k*wt)
        dQ_dbk = sinwt*sinkwt
        dQDot_dbk = w*(coswt*sinkwt+k*sinwt*coskwt)
        dQDotDot_dbk = w*w*(2*k*coswt*coskwt-(k*k+1)*sinwt*sinkwt)
        for Coord in range(NCoords(mode)):
            output[paramIndex(Coord,mode)] = fp*(PQ[Coord]*dQ_dbk + PQD[Coord]*dQDot_dbk + PQDD[Coord]*dQDotDot_dbk)
    return output
                                                
def effortJacobian(paramList):
    return SimpsonsRule(lambda t,x=paramList: FJIntegrand(*trajectory(t,x),t))

def e(i,t):  # Basis functions for constraints
    if i%2==0:
        output = cos(w*i*(t-t0))
    else:
        output = sin(w*(i+1)*(t-t0))
    return output

def CIntegrand(component,mode,Q,QDot,QDotDot,t):
    return e(mode,t)*A(component,Q,QDot,QDotDot)

def integral_constraint(component,mode,paramList):
    return SimpsonsRule(lambda t,c=component,m=mode,x=paramList: CIntegrand(c,m,*trajectory(t,x),t))

def CJIntegrand(component,mode,Q,QDot,QDotDot,t):
    output = np.empty(NParams, dtype=float)
    AQ = A_Q(component,Q,QDot,QDotDot)
    AQD = A_QDot(component,Q,QDot,QDotDot)
    AQDD = A_QDotDot(component,Q,QDot,QDotDot)
    ei = e(mode,t)
    wt = w*(t-t0)
    sinwt = sin(wt)
    coswt = cos(wt)
    for mode in range(NParamsPerPassiveCoord):
        k = mode+1
        sinkwt = sin(k*wt)
        coskwt = cos(k*wt)
        dQ_dbk = sinwt*sinkwt
        dQDot_dbk = w*(coswt*sinkwt+k*sinwt*coskwt)
        dQDotDot_dbk = w*w*(2*k*coswt*coskwt-(k*k+1)*sinwt*sinkwt)
        for Coord in range(NCoords(mode)):
            output[paramIndex(Coord,mode)] = ei*(dQ_dbk*AQ[Coord] + dQDot_dbk*AQD[Coord] + dQDotDot_dbk*AQDD[Coord])
    return output

def integral_constraintJacobian(component,mode,paramList):
    return SimpsonsRule(lambda t,c=component,m=mode,x=paramList: CJIntegrand(c,m,*trajectory(t,x),t))

def constraint(component,mode,paramList):
    dt = (t1-t0)/(NConstraintsPerPassiveCoord-1)
    t = t0+mode*dt
    return A(component,*trajectory(t,paramList))

def constraintJacobian(component,mode,paramList):
    output = np.empty(NParams, dtype=float)
    dt = (t1-t0)/(NConstraintsPerPassiveCoord-1)
    t = t0+mode*dt
    Q,QDot,QDotDot = trajectory(t,paramList)
    AQ = A_Q(component,Q,QDot,QDotDot)
    AQD = A_QDot(component,Q,QDot,QDotDot)
    AQDD = A_QDotDot(component,Q,QDot,QDotDot)
    wt = w*(t-t0)
    sinwt = sin(wt)
    coswt = cos(wt)
    for mode in range(NParamsPerPassiveCoord):
        k = mode+1
        sinkwt = sin(k*wt)
        coskwt = cos(k*wt)
        dQ_dbk = sinwt*sinkwt
        dQDot_dbk = w*(coswt*sinkwt+k*sinwt*coskwt)
        dQDotDot_dbk = w*w*(2*k*coswt*coskwt-(k*k+1)*sinwt*sinkwt)
        for Coord in range(NCoords(mode)):
            output[paramIndex(Coord,mode)] = dQ_dbk*AQ[Coord] + dQDot_dbk*AQD[Coord] + dQDotDot_dbk*AQDD[Coord]
    return output

def solve():
    conList = []
    for mode in range(NConstraintsPerPassiveCoord):
        for component in range(NPassiveCoords):
            if (useFourierConstraints):
                conList.append({'type': 'eq', 'fun': lambda x,c=component,m=mode: integral_constraint(c,m,x), 'jac': lambda x,c=component,m=mode: integral_constraintJacobian(c,m,x)})
            else:
                conList.append({'type': 'eq', 'fun': lambda x,c=component,m=mode: constraint(c,m,x), 'jac': lambda x,c=component,m=mode: constraintJacobian(c,m,x)})
    result = minimize(effort, x0=initialGuess, method='SLSQP', jac=effortJacobian, constraints=conList, options={'maxiter': maxIterations})
    return result

# ******** Main Program *********

s = solve()
print(s)
print(" ")
print("Parameter values:")
print(s.x)
print(" ")
print("effort = ",effort(s.x))
print(" ")
print("Constraint values for (component, mode):")
for mode in range(NConstraintsPerPassiveCoord):
    for component in range(NPassiveCoords):
        print("(",component,",",mode,"):",constraint(component,mode,s.x))
dt = 0.03
for i in range(int(1+(t1-t0)/dt)):
    t = t0+i*dt
    q, a, qDot, aDot, qDotDot, aDotDot = unpack(*trajectory(t,s.x))
    print("(t, q, a)=(",t,",",q[0],",",a[0],")")
