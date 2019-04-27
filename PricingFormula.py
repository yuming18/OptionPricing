from __future__ import division
from scipy.stats import norm, mvn, multivariate_normal
from tqdm import tqdm
import math
import numpy as np
import numpy.linalg as la
import copy

print (abc)

# Some math, computation and output function
def RMSE(true, price):
    true = np.array(true)
    price = np.array(price)
    n = len(price)
    return np.sqrt(np.sum((price-ture)**2)/n)

def NormalCDF(x):
    if x > 0:
        pdf = (1/np.sqrt(2*np.pi)) * np.exp(np.square(x)/-2.0)
        k = 1/(1 + 0.2316419 * x)
        cdf = 1-pdf*(0.319381530*k-0.356563782*k*k+1.781477937*k*k*k-1.821255978*k*k*k*k+1.330274429*k*k*k*k*k)
        return cdf
    else:
       return NormalCDF(-x)

def PrintTree(tree):
    for i in range(len(tree)):
        print "No.", len(tree[i]) - 1, "period."
        print tree[i]

def CholeskyDecomposition(cov,count):
    # Cholesky Decomposition
    cholesky = np.zeros((count,count))
    cholesky[0][0] = np.sqrt(cov[0][0])
    cholesky[0,1:] = cov[0][1:]/cholesky[0,0]
    for i in range(1,count-1):
        tmp2 = 0
        cholesky[i,i] = np.sqrt(cov[i,i]-np.sum(cholesky[:i,i]**2))
        for j in range(i+1,count):
            cholesky[i,j] = (cov[i,j]-np.sum(cholesky[0:i,i]*cholesky[0:i,j]))/cholesky[i,i]
    cholesky[-1][-1] = np.sqrt(cov[count-1][count-1]-np.sum(cholesky[:count-1,-1]**2))
    return cholesky

def Factorial(n):
    # return a numpy array of log 1:n
    log = list()
    for i in range(n):
        log.append(np.log(i+1))
    return np.array(log)

def Combination(nFactorialList,n,j):
    # return log(C(n,j))
    return np.sum(nFactorialList) - np.sum(nFactorialList[:j]) - np.sum(nFactorialList[:n-j])

def d1(S,K,r,q,sigma,T):
    return (np.log(S/float(K))+(r-q+(sigma**2)/2.0)*T) / (sigma*(T**0.5))

def d2(S,K,r,q,sigma,T):
    return d1(S,K,r,q,sigma,T)-(sigma*(T**0.5))

# Find the critical stock price of vanilla call option
def CriticalCallPrice(S,K,r,q,sigma,T,AcceptedError=10**-6):
    target = K
    while((abs(BS_EuroCall(target,K,r,q,sigma,T)-K)-target)>AcceptedError):
        delta = np.exp(-q*T)*norm.cdf(d1(target,K,r,q,sigma,T))
        target = target - (BS_EuroCall(target,K,r,q,sigma,T)-K-target)/delta
    return target

# Find the critical stock price of vanilla put option
def CriticalPutPrice(S,K,r,q,sigma,T,AcceptedError=10**-6):
    target = S
    while((abs(K-BS_EuroPut(target,K,r,q,sigma,T))-target)>AcceptedError):
        delta = np.exp(-q*T)*(norm.cdf(d1(target,K,r,q,sigma,T)) - 1)
        target = target - (K-BS_EuroPut(target,K,r,q,sigma,T)-target)/delta
    return target

def BS_EuroCall(S,K,r,q,sigma,T):
    D1 = d1(S,K,r,q,sigma,T)
    D2 = d2(S,K,r,q,sigma,T)
    return S * np.exp(-q*T) * norm.cdf(D1) - K*np.exp(-r*T)*norm.cdf(D2)

def BS_EuroPut(S,K,r,q,sigma,T):
    D1 = d1(S,K,r,q,sigma,T)
    D2 = d2(S,K,r,q,sigma,T)
    return K*np.exp(-r*T)*norm.cdf(-D2) - S * np.exp(-q*T) * norm.cdf(-D1)

def Combinatorial_VanillaEuroOption(S,K,r,q,sigma,T,n):
    dt = T/n
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    rnp = (np.exp((r-q)*dt) - d)/(u-d) # Risk Netural Probability

    call = 0
    put = 0
    nf = Factorial(n)# return n factorial list

    for a in range(n+1):
        call = call + np.exp(Combination(nf,n,a)+(n-a)*np.log(rnp)+a*np.log(1-rnp))*max(S*np.exp((n-a)*np.log(u)+a*np.log(d))-K,0)
        put = put + np.exp(Combination(nf,n,a)+(n-a)*np.log(rnp)+a*np.log(1-rnp))*max(K-S*np.exp((n-a)*np.log(u)+a*np.log(d)),0)

    call = np.exp(-r*T)*call
    put = np.exp(-r*T)*put

    return call, put

def CRR_EuroCall(S,K,r,q,sigma,T,n):
    dt = T/n
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    rnp = (np.exp((r-q)*dt) - d)/(u-d) # Risk Netural Probability

    call = 0
    Factorialn = Factorial(n) # return a np array of log 1:n

    for a in range(n+1):
        call = call + np.exp(Combination(Factorialn,n,a)+(n-a)*np.log(rnp)+a*np.log(1-rnp))*max(S*np.exp((n-a)*np.log(u)+a*np.log(d))-K,0)
    call = np.exp(-r*T)*call

    return call

def CRR_EuroPut(S,K,r,q,sigma,T,n):
    dt = T/n
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    rnp = (np.exp((r-q)*dt) - d)/(u-d) # Risk Netural Probability

    put = 0
    Factorialn = Factorial(n)

    for a in range(n+1):
        put = put + np.exp(Combination(Factorialn,n,a)+(n-a)*np.log(rnp)+a*np.log(1-rnp))*max(K-S*np.exp((n-a)*np.log(u)+a*np.log(d)),0)
    put = np.exp(-r*T)*put

    return put

def CRR_AmerCall(S,K,r,q,sigma,T,n):
    dt = T/n
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    rnp = (np.exp((r-q)*dt) - d)/(u-d) # Risk Netural Probability

    option = list()
    tmp = list()

    for a in range(n+1):
        tmp.append(max(S*np.float_power(u,n-a)*np.float_power(d,a)-K,0))
    option.append(tmp)

    for a in range(n):
        tmp = []
        for b in range(n-a):
            EV = S*np.float_power(u,n-b-a-1)*np.float_power(d,b) - K
            HV = np.exp(-1*r*dt)*(rnp*option[a][b]+(1-rnp)*option[a][b+1])
            tmp.append(max(EV,HV))
        option.append(tmp)

    price = option[-1][0]

    return price

def CRR_AmerPut(S,K,r,q,sigma,T,n):
    dt = T/n
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    rnp = (np.exp((r-q)*dt) - d)/(u-d) # Risk Netural Probability

    option = list()
    tmp = list()

    for a in range(n+1):
        tmp.append(max(K-S*np.float_power(u,n-a)*np.float_power(d,a),0))
    option.append(tmp)

    for a in range(n):
        tmp = []
        for b in range(n-a):
            EV = K - S*np.float_power(u,n-b-a-1)*np.float_power(d,b)
            HV = np.exp(-1*r*dt)*(rnp*option[a][b]+(1-rnp)*option[a][b+1])
            tmp.append(max(EV,HV))
        option.append(tmp)

    price = option[-1][0]

    return price

def MonteCarlo_EuroCall(S,K,r,q,sigma,T,SimulationNo=10000,RepetitionTime=20):
    average = list()

    for a in range(RepetitionTime):
        option = list()
        # Normal parameter
        mean = np.log(S)+(r-q-(sigma**2/2.0))*T
        std = sigma*np.sqrt(T)
        stock = np.random.normal(mean, std, SimulationNo)

        stock = np.exp(stock) 
        option = np.exp(-r*T)*(np.maximum(stock-K, 0))
        average.append(np.average(option))

    mid = np.average(average)
    L = np.average(average) - 2*np.std(average)
    U = np.average(average) + 2*np.std(average)

    return mid, L, U

def MonteCarlo_EuroPut(S,K,r,q,sigma,T,SimulationNo=10000,RepetitionTime=20):
    average = list()

    for a in range(RepetitionTime):
        option = list()
        # Normal parameter
        mean = np.log(S)+(r-q-(sigma**2/2.0))*T
        std = sigma*np.sqrt(T)
        stock = np.random.normal(mean, std, SimulationNo)

        stock = np.exp(stock) 
        option = np.exp(-r*T)*(np.maximum(K-stock, 0))
        average.append(np.average(option))

    mid = np.average(average)
    L = np.average(average) - 2*np.std(average)
    U = np.average(average) + 2*np.std(average)

    return mid, L, U

# FC hw5 basic
def MonteCarlo_AverEuroCall(S,K,r,q,sigma,t,T,n,Savgt,passingtime=0,SimulationNo=10000,RepetitionTime=20):

    option = list()
    dt = (T-t)/n
    passperiod = int(n*(t/(T-t)))+1

    for a in range(RepetitionTime):
        st = (np.zeros(SimulationNo)+np.log(S)).reshape(SimulationNo,1)
        dw = np.sqrt(dt)*np.random.randn(SimulationNo,n)
        for b in range(n):
            st = np.concatenate((st,st[:,-1].reshape(SimulationNo,1)+(r-q-0.5*sigma**2)*dt+sigma*dw[:,b].reshape(SimulationNo,1)),axis=1)
#            print st
        st = np.exp(st)
#        print st

        avglist = list()
        for row in st:
            rowavg = (Savgt*passperiod+np.sum(row[1:]))/(passperiod+n)
            avglist.append(rowavg)
        avglist = np.array(avglist).reshape(SimulationNo,1)

        option.append(np.average(np.exp(-r*(T-t))*np.maximum(avglist - K,0)))

    mid = np.average(option)
    L = mid - 2 * np.std(option)
    U = mid + 2 * np.std(option)

    return mid, L, U

# Principle of Financial Calculation hw1
def CallOnCall(S,K1,K2,r,q,sigma,t,T):

    def FindS(S,K1,K2,r,q,sigma,T,AcceptedError=10**-6):
        target = S
        while(abs(BS_EuroCall(target,K2,r,q,sigma,T) - K1) > AcceptedError):
            delta = norm.cdf(d1(target,K2,r,q,sigma,T))
            target = target - 0.1 * (BS_EuroCall(target,K2,r,q,sigma,T)-K1)/delta
        return target

    CriticalPrice = FindS(S,K1,K2,r,q,sigma,T-t)
    print "The critical stock price of compound option is:", CriticalPrice

    # Multivariate Normal cdf parameter
    a1 = d1(S,CriticalPrice,r,q,sigma,t)
    a2 = d2(S,CriticalPrice,r,q,sigma,t)
    b1 = d1(S,K2,r,q,sigma,T)
    b2 = d2(S,K2,r,q,sigma,T)
#    print a1, a2, b1, b2
    cor = np.sqrt(t/T)
    mu = np.array([0.0,0.0])
    var = np.array([[1,cor*1*1],[cor*1*1,1]])
    low = np.array([-10,-10])
    up1 = np.array([a1,b1])
    up2 = np.array([a2,b2])
    
    # Multivariate Normal cdf
    p1 = multivariate_normal.cdf(up1,mu,var)
    p2 = multivariate_normal.cdf(up2,mu,var)
#    p1,i = mvn.mvnun(low,up1,mu,var)
#    p2,i = mvn.mvnun(low,up2,mu,var) 

    price = S*np.exp(-q*T)*p1 - K2*np.exp(-r*T)*p2 - K1*np.exp(-r*t)*norm.cdf(a2)
    
    print "Compound option premium:"
    return price

# Principle of Financial Calculation hw2
def BermudanPut2(S,K,r,q,sigma,T):
    
    AcceptedError = 0.0001

    t1 = T/2.0
    CriticalPrice = CriticalPutPrice(S,K,r,q,sigma,t1,AcceptedError)
#    print "Early Exercise Critical Price at T/2 is:", CriticalPrice

    a2 = d2(S,CriticalPrice,r,q,sigma,t1)
    p1 = norm.cdf(-a2)

    a1 = d1(S,CriticalPrice,r,q,sigma,t1)
    p2 = norm.cdf(-a1)

    b2 = d2(S,K,r,q,sigma,T)
    cor = -np.float_power(2,-0.5)
    cov = np.array([[1,cor],[cor,1]])
    p3 = multivariate_normal.cdf(np.array([a2,-b2]),np.array([0,0]),cov)

    b1 = d1(S,K,r,q,sigma,T)
    p4 = multivariate_normal.cdf(np.array([a1,-b1]),np.array([0,0]),cov)

    price = K*np.exp(-r*t1)*p1 - S*p2 + K*np.exp(-r*T)*p3 - S*p4

    return price

# Principle of Financial Calculation hw2
def BermudanPut3(S,K,r,q,sigma,T):
    
    AcceptedError = 10**-6

    def FindTPrice(S,K,r,q,sigma,T,AcceptedError = 0.0001):
        a = AcceptedError
        b = K
        c = (a+b)/2
        while(abs(b - c) > AcceptedError):
            if(np.sign(K - b - BermudanPut2(b,K,r,q,sigma,T)) * np.sign(K - c - BermudanPut2(c,K,r,q,sigma,T)) <= 0):
                a = c
            else:
                b = c
            c = (a+b)/2
        return c

    t1 = T/3.0
    t2 = (2*T)/3.0

    TPrice = FindTPrice(S,K,r,q,sigma,t2)
    print "Early Exercise Critical Price at T/3 is:", TPrice
    T2Price = CriticalPutPrice(S,K,r,q,sigma,t1,AcceptedError)
    print "Early Exercise Critical Price at 2T/3 is:", T2Price

    a2 = d2(S,TPrice,r,q,sigma,t1)
    p1 = norm.cdf(-a2)

    a1 = d1(S,TPrice,r,q,sigma,t1)
    p2 = norm.cdf(-a1)

    b2 = d2(S,T2Price,r,q,sigma,t2)
    cor12 = -np.float_power(2,-0.5)
    cov = np.array([[1,cor12*1*1],[cor12*1*1,1]])
    p3 = multivariate_normal.cdf(np.array([a2,-b2]),np.array([0,0]),cov)

    b1 = d1(S,T2Price,r,q,sigma,t2)
    p4 = multivariate_normal.cdf(np.array([a1,-b1]),np.array([0,0]),cov)

    c2 = d2(S,K,r,q,sigma,T)
    cor12 = np.float_power(2,-0.5)
    cor13 = -np.float_power(3,-0.5)
    cor23 = -np.float_power(2/3.0,0.5)
    cov = np.array([[1,cor12,cor13],[cor12,1,cor23],[cor13,cor23,1]])
    p5 = multivariate_normal.cdf(np.array([a2,b2,-c2]),np.array([0,0,0]),cov)

    c1 = d1(S,K,r,q,sigma,T)
    p6 = multivariate_normal.cdf(np.array([a1,b1,-c1]),np.array([0,0,0]),cov)

    price = K*np.exp(-r*t1)*p1 - S*p2 + K*np.exp(-r*t2)*p3 - S*p4 + K*np.exp(-r*T)*p5 - S*p6

    print "Bermuda put option with 3 early exercise price is:"
    return price

# Principle of Financial Calculation hw3
def Ju1998(S,K,r,q,sigma,T,split):

    def I(t1,t2,x,y,z,phi,v,r,q,sigma):
        z1 = (r-q-z+phi*(sigma**2)/2)/sigma
        z2 = np.log(x/y)/sigma
        z3 = (z1**2+2*v)**0.5
        if t1 == 0:
            if z2 > 0:
                return np.exp(-v*t1) - np.exp(-v*t2)*norm.cdf(z1*(t2**0.5)+z2/(t2**0.5)) + (1/2)*(z1/z3 + 1)*np.exp(z2*(z3-z1))*(norm.cdf(z3*(t2**0.5)+z2/(t2**0.5)) - 1) + (1/2)*(z1/z3 - 1)*np.exp(-z2*(z3+z1))*(norm.cdf(z3*(t2**0.5)-z2/(t2**0.5)))
            elif z2 == 0:
                return np.exp(-v*t1)*0.5 - np.exp(-v*t2)*norm.cdf(z1*(t2**0.5)+z2/(t2**0.5)) + (1/2)*(z1/z3 + 1)*np.exp(z2*(z3-z1))*(norm.cdf(z3*(t2**0.5)+z2/(t2**0.5)) - 0.5) + (1/2)*(z1/z3 - 1)*np.exp(-z2*(z3+z1))*(norm.cdf(z3*(t2**0.5)-z2/(t2**0.5)) - 0.5)
            else:
                return np.exp(-v*t1)*0 - np.exp(-v*t2)*norm.cdf(z1*(t2**0.5)+z2/(t2**0.5)) + (1/2)*(z1/z3 + 1)*np.exp(z2*(z3-z1))*(norm.cdf(z3*(t2**0.5)+z2/(t2**0.5))) + (1/2)*(z1/z3 - 1)*np.exp(-z2*(z3+z1))*(norm.cdf(z3*(t2**0.5)-z2/(t2**0.5)) - 1)
        else:
            return np.exp(-v*t1)*norm.cdf(z1*(t1**0.5)+z2/(t1**0.5)) - np.exp(-v*t2)*norm.cdf(z1*(t2**0.5)+z2/(t2**0.5)) + (1/2)*(z1/z3 + 1)*np.exp(z2*(z3-z1))*(norm.cdf(z3*(t2**0.5)+z2/(t2**0.5)) - norm.cdf(z3*(t1**0.5)+z2/(t1**0.5))) + (1/2)*(z1/z3 - 1)*np.exp(-z2*(z3+z1))*(norm.cdf(z3*(t2**0.5)-z2/(t2**0.5)) - norm.cdf(z3*(t1**0.5)-z2/(t1**0.5)))

    def Is(t1,t2,x,y,z,phi,v,r,q,sigma):
        if t1 != 0:
            z1 = (r-q-z+(phi*sigma**2)/2)/sigma
            z2 = np.log(x/y)/sigma
            z3 = np.sqrt(z1**2+2*v)
        
            one = ((np.exp(-v*t1)/np.sqrt(t1))*norm.pdf(z1*np.sqrt(t1)+(z2/np.sqrt(t1)))-(np.exp(-v*t2)/np.sqrt(t2))*norm.pdf(z1*np.sqrt(t2)+(z2/np.sqrt(t2))))/(sigma*x)
            two = ((z1/z3)+1)*np.exp(z2*(z3-z1))*(norm.cdf(z3*np.sqrt(t2)+z2/np.sqrt(t2))-norm.cdf(z3*np.sqrt(t1)+z2/np.sqrt(t1)))*(z3-z1)/(2*sigma*x)
            three = ((z1/z3)+1)*np.exp(z2*(z3-z1))*(norm.pdf(z3*np.sqrt(t2)+z2/np.sqrt(t2))/np.sqrt(t2)-norm.pdf(z3*np.sqrt(t1)+z2/np.sqrt(t1))/np.sqrt(t1))/(2*sigma*x)
            four = ((z1/z3)-1)*np.exp(-z2*(z3+z1))*(norm.cdf(z3*np.sqrt(t2)-z2/np.sqrt(t2))-norm.cdf(z3*np.sqrt(t1)-z2/np.sqrt(t1)))*(z3+z1)/(2*sigma*x)
            five = ((z1/z3)-1)*np.exp(-z2*(z3+z1))*(norm.pdf(z3*np.sqrt(t2)-z2/np.sqrt(t2))/np.sqrt(t2)-norm.pdf(z3*np.sqrt(t1)-z2/np.sqrt(t1))/np.sqrt(t1))/(2*sigma*x)
            return one+two+three-four-five
        else:
            delta = 10**-9
            return (I(t1,t2,x+delta,y,z,phi,v,r,q,sigma)-I(t1,t2,x,y,z,phi,v,r,q,sigma))/delta
        
    def VM11(B11,b11,K,r,q,sigma,T):
        S = B11
        left = K - S
        right = BS_EuroPut(S,K,r,q,sigma,T)+K*(1-np.exp(-r*T))-S*(1-np.exp(-q*T))-K*I(0,T,S,S,b11,-1,r,r,q,sigma)+S*I(0,T,S,S,b11,1,q,r,q,sigma)
        return left - right

    def HC11(B11,b11,K,r,q,sigma,T):
        S = B11
        left = -1
        right = -1*np.exp(-q*T)*norm.cdf(-1*d1(S,K,r,q,sigma,T))-(1-np.exp(-q*T))-K*Is(0,T,S,S,b11,-1,r,r,q,sigma)+I(0,T,S,S,b11,1,q,r,q,sigma)+S*Is(0,T,S,S,b11,1,q,r,q,sigma)
        return left - right

    def VM21(B21,b21,K,r,q,sigma,T):
        S = B21*np.exp(b21*T/2)
        left = K - S
        right = BS_EuroPut(S,K,r,q,sigma,T/2)+K*(1-np.exp(-r*T/2))-S*(1-np.exp(-q*T/2))-K*I(0,T/2,S,S,b21,-1,r,r,q,sigma)+S*I(0,T/2,S,S,b21,1,q,r,q,sigma)
        return left - right

    def HC21(B21,b21,K,r,q,sigma,T):
        S = B21*np.exp(b21*T/2)
        left = -1 
        right = -1*np.exp(-q*T/2)*norm.cdf(-1*d1(S,K,r,q,sigma,T/2))-(1-np.exp(-q*T/2))-K*Is(0,T/2,S,S,b21,-1,r,r,q,sigma)+I(0,T/2,S,S,b21,1,q,r,q,sigma)+S*Is(0,T/2,S,S,b21,1,q,r,q,sigma)
        return left - right

    def VM22(B22,b22,B21,b21,K,r,q,sigma,T):
        S = B22
        left = K - S
        euro = BS_EuroPut(S,K,r,q,sigma,T)
        right1 = K*(1-np.exp(-r*T))-S*(1-np.exp(-q*T))
        right2 = -K*I(0,T/2,S,S,b22,-1,r,r,q,sigma)+S*I(0,T/2,S,S,b22,1,q,r,q,sigma)
        right3 = -K*I(T/2,T,S,B21,b21,-1,r,r,q,sigma)+S*I(T/2,T,S,B21,b21,1,q,r,q,sigma)
        right = euro + right1 + right2 + right3
        return left - right

    def HC22(B22,b22,B21,b21,K,r,q,sigma,T):
        S = B22
        left = -1
        right = -np.exp(-q*T)*norm.cdf(-1*d1(S,K,r,q,sigma,T))-(1-np.exp(-q*T))-K*Is(0,T/2,S,S,b22,-1,r,r,q,sigma)+I(0,T/2,S,S,b22,1,q,r,q,sigma)+S*Is(0,T/2,S,S,b22,1,q,r,q,sigma)-K*Is(T/2,T,S,B21,b21,-1,r,r,q,sigma)+I(T/2,T,S,B21,b21,1,q,r,q,sigma)+S*Is(T/2,T,S,B21,b21,1,q,r,q,sigma)
        return left - right

    def VM31(B31,b31,K,r,q,sigma,T):
        S = B31*np.exp(b31*2*T/3)
        left = K - S
        right = BS_EuroPut(S,K,r,q,sigma,T/3)+K*(1-np.exp(-r*T/3))-S*(1-np.exp(-q*T/3))-K*I(0,T/3,S,S,b31,-1,r,r,q,sigma)+S*I(0,T/3,S,S,b31,1,q,r,q,sigma)  
        return left - right

    def HC31(B31,b31,K,r,q,sigma,T):
        S = B31*np.exp(b31*2*T/3)
        left = -1
        right = -1*np.exp(-q*T/3)*norm.cdf(-1*d1(S,K,r,q,sigma,T/3))-(1- np.exp(-q*T/3))-K*Is(0,T/3,S,S,b31,-1,r,r,q,sigma)+I(0,T/3,S,S,b31,1,q,r,q,sigma)+S*Is(0,T/3,S,S,b31,1,q,r,q,sigma)
        return left -right

    def VM32(B32,b32,B31,b31,K,r,q,sigma,T):
        S = B32*np.exp(b32*T/3)
        S2 = B31*np.exp(b31*T/3)
        left = K - S
        euro = BS_EuroPut(S,K,r,q,sigma,2*T/3)
        right1 = K*(1-np.exp(-r*2*T/3))-S*(1-np.exp(-q*2*T/3))
        right2 = -K*I(0,T/3,S,S,b32,-1,r,r,q,sigma)+S*I(0,T/3,S,S,b32,1,q,r,q,sigma)
        right3 = -K*I(T/3,2*T/3,S,S2,b31,-1,r,r,q,sigma)+S*I(T/3,2*T/3,S,S2,b31,1,q,r,q,sigma)
        right = euro + right1 + right2 + right3
        return left - right

    def HC32(B32,b32,B31,b31,K,r,q,sigma,T):
        S = B32*np.exp(b32*T/3)
        S2 = B31*np.exp(b31*T/3)
        left = -1
        right1 = -1*np.exp(-q*2*T/3)*norm.cdf(-1*d1(S,K,r,q,sigma,2*T/3))-(1- np.exp(-q*2*T/3))
        right2 = -K*Is(0,T/3,S,S,b32,-1,r,r,q,sigma)+I(0,T/3,S,S,b32,1,q,r,q,sigma)+S*Is(0,T/3,S,S,b32,1,q,r,q,sigma)
        right3 = -K*Is(T/3,2*T/3,S,S2,b31,-1,r,r,q,sigma)+I(T/3,2*T/3,S,S2,b31,1,q,r,q,sigma)+S*Is(T/3,2*T/3,S,S2,b31,1,q,r,q,sigma)
        right = right1 + right2 + right3

        return left - right
                
    def VM33(B33,b33,B32,b32,B31,b31,K,r,q,sigma,T):
        S = B33
        left = K - S
        right = BS_EuroPut(B33,K,r,q,sigma,T)+K*(1-np.exp(-r*T))-S*(1-np.exp(-q*T))-K*I(0,T/3,S,S,b33,-1,r,r,q,sigma)+S*I(0,T/3,S,S,b33,1,q,r,q,sigma)-K*I(T/3,2*T/3,S,B32,b32,-1,r,r,q,sigma)+S*I(T/3,2*T/3,S,B32,b32,1,q,r,q,sigma)-K*I(2*T/3,T,S,B31,b31,-1,r,r,q,sigma)+S*I(2*T/3,T,S,B31,b31,1,q,r,q,sigma)
        return left - right

    def HC33(B33,b33,B32,b32,B31,b31,K,r,q,sigma,T):
        S = B33
        left = -1
        right = -1*np.exp(-q*T)*norm.cdf(-1*d1(S,K,r,q,sigma,T))-(1- np.exp(-q*T))-K*Is(0,T/3,S,S,b33,-1,r,r,q,sigma)+I(0,T/3,S,S,b33,1,q,r,q,sigma)+S*Is(0,T/3,S,S,b33,1,q,r,q,sigma)-K*Is(T/3,2*T/3,S,B32,b32,-1,r,r,q,sigma)+I(T/3,2*T/3,S,B32,b32,1,q,r,q,sigma)+S*Is(T/3,2*T/3,S,B32,b32,1,q,r,q,sigma)-K*Is(2*T/3,T,S,B31,b31,-1,r,r,q,sigma)+I(2*T/3,T,S,B31,b31,1,q,r,q,sigma)+S*Is(2*T/3,T,S,B31,b31,1,q,r,q,sigma)
        return left - right

    def FindInitialB11(S,X,r,q,sigma,T):
        N = 2*(r-q)/(sigma**2)
        M = 2*r/(sigma**2)
        K = 1-np.exp(-r*T)
        q1 = (-(N-1)-(((N-1)**2)+((4*M)/K))**0.5)/2
        def LHS(S,X):
            return X-S
        def RHS(S,X,r,q,sigma,T,N,M,K,q1):
             return BS_EuroPut(S,X,r,q,sigma,T)-((1-np.exp(-q*T)*norm.cdf(-d1(S,X,r,q,sigma,T)))*S/q1)
        def b1(S,X,r,q,sigma,T,N,M,K,q1):
            return np.exp(-q*T)*norm.cdf(-d1(S,X,r,q,sigma,T))*(1-(1/q1)) - (1+((np.exp(-q*T)*norm.pdf(-d1(S,X,r,q,sigma,T)))/(sigma*(T**0.5))))/q1
        init = S
        while(abs(LHS(init,X)-RHS(init,X,r,q,sigma,T,N,M,K,q1))/X > 0.00001):
#            print init
            init = (-X + RHS(init,X,r,q,sigma,T,N,M,K,q1)-b1(init,X,r,q,sigma,T,N,M,K,q1)*init)/(-1-b1(init,X,r,q,sigma,T,N,M,K,q1))
        return init

#   Jac adjust
    def JacJump(J11,J12,J21,J22,jump):
#        print "jump"
        J11 = min(J11,jump)
        J12 = min(J12,jump)
        J21 = min(J21,jump)
        J22 = min(J22,jump)
        return np.array([[J11,J12],[J21,J22]])

    def BbJump(Bb,B,b,delta):
        if abs(Bb[0]-B) > delta*1000:
            Bb[0] = (B + Bb[0])/2
        if abs(Bb[1]-b) > delta:
            Bb[1] = (b + Bb[1])/2
        return Bb
    
    error = 10**-6
    delta = error/100
    jump = -10**-6
    
#   Two variable Newton: x = x - [f,g]'*J(f,g)^-1 of B11 and b11
    B11 = FindInitialB11(S,K,r,q,sigma,T)
    b11 = 0.001
    print "B11 Initiate =", B11

    while(abs(VM11(B11,b11,K,r,q,sigma,T))>error or abs(HC11(B11,b11,K,r,q,sigma,T))>error):
        Bb = np.array([B11,b11])
        vm = VM11(B11,b11,K,r,q,sigma,T)
        hc = HC11(B11,b11,K,r,q,sigma,T)
        J11 = (VM11(B11+delta,b11,K,r,q,sigma,T)-vm)/delta
        J12 = (VM11(B11,b11+delta,K,r,q,sigma,T)-vm)/delta
        J21 = (HC11(B11+delta,b11,K,r,q,sigma,T)-hc)/delta
        J22 = (HC11(B11,b11+delta,K,r,q,sigma,T)-hc)/delta
#        Jac = np.array([[J11,J12],[J21,J22]])
        Jac = JacJump(J11,J12,J21,J22,jump)
        Bb = Bb - np.matmul(la.inv(Jac), np.array([vm,hc]))
        Bb = BbJump(Bb,B11,b11,delta)
#        print Bb
        B11 = Bb[0]
        b11 = Bb[1]
    print "B11 optimal =", B11
    print "b11 optimal =", b11

    if split == 1:
        if S > B11:
            return BS_EuroPut(S,K,r,q,sigma,T) + K*(1-np.exp(-r*T)) - S*(1-np.exp(-q*T)) - K*I(0,T,S,B11,b11,-1,r,r,q,sigma) + S*I(0,T,S,B11,b11,1,q,r,q,sigma)
        else:
            return K-S

#   Two variable Newton: x = x - [f,g]'*J(f,g)^-1 of B21 and b21
    B21 = B11
    b21 = b11

    while(abs(VM21(B21,b21,K,r,q,sigma,T))>error or abs(HC21(B21,b21,K,r,q,sigma,T))>error):
        Bb = np.array([B21,b21])
        vm = VM21(B21,b21,K,r,q,sigma,T)
        hc = HC21(B21,b21,K,r,q,sigma,T)
        J11 = (VM21(B21+delta,b21,K,r,q,sigma,T)-vm)/delta
        J12 = (VM21(B21,b21+delta,K,r,q,sigma,T)-vm)/delta
        J21 = (HC21(B21+delta,b21,K,r,q,sigma,T)-hc)/delta
        J22 = (HC21(B21,b21+delta,K,r,q,sigma,T)-hc)/delta
#        Jac = np.array([[J11,J12],[J21,J22]])
        Jac = JacJump(J11,J12,J21,J22,jump)
        Bb = Bb - np.matmul(la.inv(Jac), np.array([vm,hc]))
        Bb = BbJump(Bb,B21,b21,delta)
#        print "Bb =", Bb
        B21 = Bb[0]
        b21 = Bb[1]

    print "B21 optimal =", B21
    print "b21 optimal =", b21

#   Two variable Newton: x = x - [f,g]'*J(f,g)^-1 of B22 and b22
    B22 = B21
    b22 = b21
    
    while(abs(VM22(B22,b22,B21,b21,K,r,q,sigma,T))>error or abs(HC22(B22,b22,B21,b21,K,r,q,sigma,T))>error):
#        delta = error/100
        Bb = np.array([B22,b22])
        vm = VM22(B22,b22,B21,b21,K,r,q,sigma,T)
        hc = HC22(B22,b22,B21,b21,K,r,q,sigma,T)
        J11 = (VM22(B22+delta,b22,B21,b21,K,r,q,sigma,T)-vm)/delta
        J12 = (VM22(B22,b22+delta,B21,b21,K,r,q,sigma,T)-vm)/delta
        J21 = (HC22(B22+delta,b22,B21,b21,K,r,q,sigma,T)-hc)/delta
        J22 = (HC22(B22,b22+delta,B21,b21,K,r,q,sigma,T)-hc)/delta
#        Jac = np.array([[J11,J12],[J21,J22]])
        Jac = JacJump(J11,J12,J21,J22,jump)
        Bb = Bb - np.matmul(la.inv(Jac), np.array([vm,hc]))
        Bb = BbJump(Bb,B22,b22,delta)
#        print "Bb =", Bb
        B22 = Bb[0]
        b22 = Bb[1]

    print "B22 optimal =", B22
    print "b22 optimal =", b22

    if split == 2:
        if S > B22:
            return BS_EuroPut(S,K,r,q,sigma,T) + K*(1-np.exp(-r*T)) - S*(1-np.exp(-q*T)) - K*I(0,T/2,S,B22,b22,-1,r,r,q,sigma) + S*I(0,T/2,S,B22,b22,1,q,r,q,sigma) - K*I(T/2,T,S,B21,b21,-1,r,r,q,sigma) + S*I(T/2,T,S,B21,b21,1,q,r,q,sigma)
        else:
            return K-S

#   Two variable Newton: x = x - [f,g]'*J(f,g)^-1 of B31 and b31
    B31 = B21
    b31 = b21
    
    while(abs(VM31(B31,b31,K,r,q,sigma,T))>error or abs(HC31(B31,b31,K,r,q,sigma,T))>error):
#        delta = error/10
        Bb = np.array([B31,b31])
        vm = VM31(B31,b31,K,r,q,sigma,T)
        hc = HC31(B31,b31,K,r,q,sigma,T)
        J11 = (VM31(B31+delta,b31,K,r,q,sigma,T)-vm)/delta
        J12 = (VM31(B31,b31+delta,K,r,q,sigma,T)-vm)/delta
        J21 = (HC31(B31+delta,b31,K,r,q,sigma,T)-hc)/delta
        J22 = (HC31(B31,b31+delta,K,r,q,sigma,T)-hc)/delta
#        Jac = np.array([[J11,J12],[J21,J22]])
        Jac = JacJump(J11,J12,J21,J22,jump)
        Bb = Bb - np.matmul(la.inv(Jac), np.array([vm,hc]))
        Bb = BbJump(Bb,B31,b31,delta)
#        print "Bb =", Bb
        B31 = Bb[0]
        b31 = Bb[1]

    print "B31 optimal =", B31
    print "b31 optimal =", b31

#   Two variable Newton: x = x - [f,g]'*J(f,g)^-1 of B32 and b32
    B32 = B31
    b32 = b31
    
    while(abs(VM32(B32,b32,B31,b31,K,r,q,sigma,T))>error or abs(HC32(B32,b32,B31,b31,K,r,q,sigma,T))>error):
        delta = error/100
        Bb = np.array([B32,b32])
        vm = VM32(B32,b32,B31,b31,K,r,q,sigma,T)
        hc = HC32(B32,b32,B31,b31,K,r,q,sigma,T)
        J11 = (VM32(B32+delta,b32,B31,b31,K,r,q,sigma,T)-vm)/delta
        J12 = (VM32(B32,b32+delta,B31,b31,K,r,q,sigma,T)-vm)/delta
        J21 = (HC32(B32+delta,b32,B31,b31,K,r,q,sigma,T)-hc)/delta
        J22 = (HC32(B32,b32+delta,B31,b31,K,r,q,sigma,T)-hc)/delta
#        Jac = np.array([[J11,J12],[J21,J22]])
        Jac = JacJump(J11,J12,J21,J22,jump)        
        Bb = Bb - np.matmul(la.inv(Jac), np.array([vm,hc]))
        Bb =  BbJump(Bb,B32,b32,delta)
#        print "Bb =", Bb
        B32 = Bb[0]
        b32 = Bb[1]

    print "B32 optimal =", B32
    print "b32 optimal =", b32

#   Two variable Newton: x = x - [f,g]'*J(f,g)^-1 of B33 and b33
    B33 = B32
    b33 = b32
    
    while(abs(VM33(B33,b33,B32,b32,B31,b31,K,r,q,sigma,T))>error or abs(HC33(B33,b33,B32,b32,B31,b31,K,r,q,sigma,T))>error):
        delta = error/10
        Bb = np.array([B33,b33])
        vm = VM33(B33,b33,B32,b32,B31,b31,K,r,q,sigma,T)
        hc = HC33(B33,b33,B32,b32,B31,b31,K,r,q,sigma,T)
        J11 = (VM33(B33+delta,b33,B32,b32,B31,b31,K,r,q,sigma,T)-vm)/delta
        J12 = (VM33(B33,b33+delta,B32,b32,B31,b31,K,r,q,sigma,T)-vm)/delta
        J21 = (HC33(B33+delta,b33,B32,b32,B31,b31,K,r,q,sigma,T)-hc)/delta
        J22 = (HC33(B33,b33+delta,B32,b32,B31,b31,K,r,q,sigma,T)-hc)/delta
#        Jac = np.array([[J11,J12],[J21,J22]])
        Jac = JacJump(J11,J12,J21,J22,jump)        
        Bb = Bb - np.matmul(la.inv(Jac), np.array([vm,hc]))
        Bb = BbJump(Bb,B33,b33,delta)
#        print "Bb =", Bb
        B33 = Bb[0]
        b33 = Bb[1]

    print "B33 optimal =", B33
    print "b33 optimal =", b33
    
#    B31 = 50.87
#    b31 = 0.075
#    B32 = 53.705
#    b32 = 0.045
#    B33 = 54.452
#    b33 = 0.029

    if S > B33:
        return BS_EuroPut(S,K,r,q,sigma,T) + K*(1-np.exp(-r*T)) - S*(1-np.exp(-q*T)) - K*I(0,T/3,S,B33,b33,-1,r,r,q,sigma) + S*I(0,T/3,S,B33,b33,1,q,r,q,sigma) - K*I(T/3,2*T/3,S,B32,b32,-1,r,r,q,sigma) + S*I(T/3,2*T/3,S,B32,b32,1,q,r,q,sigma) - K*I(2*T/3,T,S,B31,b31,-1,r,r,q,sigma) + S*I(2*T/3,T,S,B31,b31,1,q,r,q,sigma)
    else:
        return K-S

# Financial Computation hw1 basic requirement
def CustomizedOption(S,K1,K2,K3,K4,r,q,sigma,T):
    d11 = d1(S,K1,r,q,sigma,T)
    d12 = d2(S,K1,r,q,sigma,T)
    d21 = d1(S,K2,r,q,sigma,T)
    d22 = d2(S,K2,r,q,sigma,T)
    d31 = d1(S,K3,r,q,sigma,T)
    d32 = d2(S,K3,r,q,sigma,T)
    d41 = d1(S,K4,r,q,sigma,T)
    d42 = d2(S,K4,r,q,sigma,T)

    RA = norm.cdf(d11)-norm.cdf(d21)
    QA = norm.cdf(d12)-norm.cdf(d22)
    QB = norm.cdf(d22)-norm.cdf(d32)
    RC = norm.cdf(d31)-norm.cdf(d41)
    QC = norm.cdf(d32)-norm.cdf(d42)

    return S*np.exp(-q*T)*RA - K1*np.exp(-r*T)*QA + (K2-K1)*np.exp(-r*T)*QB + ((K1-K2)/(K4-K3))*(S*np.exp(-q*T)*RC - K4*np.exp(-r*T)*QC)

# Financial Computation hw1 bouns
def MonteCarlo_CustomizedCall(S,K1,K2,K3,K4,r,q,sigma,T,SimulationNo,RepetitionTime):
    average = list()

    for a in range(RepetitionTime):
        option = list()
        # Normal parameter
        mean = np.log(S)+(r-q-(sigma**2/2.0))*T
        std = sigma*np.sqrt(T)
        price = np.random.normal(mean, std, SimulationNo)
        price = [np.exp(x) for x in price]

        for p in price:
            if K2 > p and p > K1:
                option.append((p-K1)*np.exp(-r*T))
            elif(K3 > p and p > K2):
                option.append((K2-K1)*np.exp(-r*T))
            elif(K4 > p and p > K3):
                option.append(((K1-K2)/(K4-K3))*(p-K4)*np.exp(-r*T))
            else:
                option.append(0)

        option = np.array(option)
        average.append(np.average(option))

    U = np.average(average) + 2*np.std(average)
    L = np.average(average) - 2*np.std(average)

    return U, L

# Principle of Financial Calculation hw4
def BBS_EuroCall(S,K,r,q,sigma,T,n):
    dt = T/n
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    rnp = (np.exp((r-q)*dt) - d)/(u-d) # Risk Netural Probability
    stock = []
    option = []
    for a in range(n):
        stock.append(S*np.float_power(u,n-(a+1))*np.float_power(d,a))
    for s in stock:
        option.append(BS_EuroCall(s,K,r,q,sigma,dt))
    for a in range(n-1):
        for b in range((n-1)-a):
            option[b] = np.exp(-1*r*dt) * (rnp * option[b] + (1-rnp) * option[b+1])
    return option[0]

def EFB_EuroCall(S,K,r,q,sigma,T,n): #Extrapolated Flexible Binomial
    dt = T/n
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    ita = int(round((np.log(K/S)-n*np.log(d))/(np.log(u/d))))
    lam = (np.log(K/S)-(2*ita-n)*sigma*np.sqrt(dt))/(n*(sigma**2)*dt)

    u = np.exp(sigma*np.sqrt(dt)+lam*(sigma**2)*dt)
    d = np.exp(-sigma*np.sqrt(dt)+lam*(sigma**2)*dt)
    rnp = (np.exp((r-q)*dt) - d)/(u-d)
#    print S*np.float_power(u,ita)*np.float_power(d,n-ita)

    stock = []
    option = []
    for a in range(n+1):
        stock.append(S*np.float_power(u,n-a)*np.float_power(d,a))
    option = [s-K if s > K else 0 for s in stock]
    for a in range(n):
        for b in range(n-a):
            option[b] = np.exp(-1*r*dt) * (rnp * option[b] + (1-rnp) * option[b+1])
    return option[0]
    
def GCRRXPC_EuroCall(S,K,r,q,sigma,T,n):
    dt = T/n
    a = (np.log(K/S))/(sigma*np.sqrt(n*T))
    lam = a + np.sqrt(a**2+1)

    u = np.exp(lam*sigma*np.sqrt(dt))
    d = np.exp((-sigma*np.sqrt(dt))/lam)
    rnp = (np.exp((r-q)*dt) - d)/(u-d)

    stock = []
    option = []
    for a in range(n+1):
        stock.append(S*np.float_power(u,n-a)*np.float_power(d,a))
    option = [s-K if s > K else 0 for s in stock]
    for a in range(n):
        for b in range(n-a):
            option[b] = np.exp(-1*r*dt) * (rnp * option[b] + (1-rnp) * option[b+1])
    return option[0]

def LEIS_EuroCall(S,K,r,q,sigma,T,n):
    dt = T/n
    z1 = d1(S,K,r,q,sigma,T)
    z2 = d2(S,K,r,q,sigma,T)
    p1 = 0.5+np.sqrt((0.25-0.25*np.exp(-np.square(z1/(n+(1/3)))*(n+(1/6)))))
    p2 = 0.5+np.sqrt((0.25-0.25*np.exp(-np.square(z2/(n+(1/3)))*(n+(1/6)))))
    u = (np.exp(r*dt)*p1)/p2
    d = (np.exp(r*dt)-p2*u)/(1-p2)

    rnp = (np.exp((r-q)*dt) - d)/(u-d)
    stock = []
    option = []
    for a in range(n+1):
        stock.append(S*np.float_power(u,n-a)*np.float_power(d,a))
    option = [s-K if s > K else 0 for s in stock]
    for a in range(n):
        for b in range(n-a):
            option[b] = np.exp(-1*r*dt) * (rnp * option[b] + (1-rnp) * option[b+1])
    return option[0]
    
def HZ_EuroCall(S,K,r,q,sigma,T,n):
    
    def integrate(K,s,dx):
        y = np.log(K/s)
        if(y < -dx):
            area = s*np.exp(dx)-s*np.exp(-dx)-2*dx*K
        elif(-dx < y and y < dx):
            area = s*np.exp(dx)-dx*K-K+y*K
        else:
            area = 0
        return area

    dt = T/n
    u = np.exp((r-sigma**2/2)*dt+sigma*np.sqrt(dt))
    d = np.exp((r-sigma**2/2)*dt-sigma*np.sqrt(dt))
    rnp = 0.5

    dx = sigma*np.sqrt(dt)
    stock = []
    option = []

    for a in range(n+1):
        stock.append(S*np.float_power(u,n-a)*np.float_power(d,a))
    option = [integrate(K,s,dx)/(2*dx) for s in stock]

    for a in range(n):
        for b in range(n-a):
            option[b] = np.exp(-1*r*dt) * (rnp * option[b] + (1-rnp) * option[b+1])
    return option[0]

# Principle of Financial Calculation hw5
def AsianCall(S,K,r,q,sigma,T,n,m,Amer=True):
    t = T/n
    u = np.exp(sigma*np.sqrt(t))
    d = 1/u
    rnp = (np.exp((r-q)*t) - d)/(u-d) # Risk Netural Probability
#    print u, d, rnp

    stock = []
    savg = list()
    option = list()
    tmp = list()
    tmp2 = list()

    for a in range(n+1):
        for b in range(a+1):
            tmp.append(S*np.float_power(u,a-b)*np.float_power(d,b))
        stock.append(tmp)
        tmp = []
#    PrintTree(stock)

    for a in range(n+1):
        for b in range(a+1):
            if a == 0:
                avg = [stock[a][b]]*m
            else:
                if b == 0:
                    avg = [(stock[a][b] + savg[-1][b][0]*a)/(a+1)]*m
                elif b == a:
                    avg = [(stock[a][b] + savg[-1][b-1][0]*a)/(a+1)]*m
                else:
                    up = (max(savg[-1][b-1])*a + stock[a][b])/(a+1)
                    down = (min(savg[-1][b])*a + stock[a][b])/(a+1)
                    # linear space
                    # avg = list(np.linspace(down, up, m))

                    # log space 
                    avg = list(np.exp(np.linspace(np.log(down),np.log(up),m)))
            tmp.append(avg)
        savg.append(tmp)
        tmp = []
#    PrintTree(savg)

    for a in range(n+1):
        tmp.append([max(savg[-1][a][i]-K,0) for i in range(m)])
    option.append(tmp)
#    PrintTree(option)

    tmp = []
    for a in range(n):
        for b in range(n-a):
            for i in range(m):
                cu = 0
                cd = 0
                upprice = (savg[n-a-1][b][i] * (n-a) + stock[n-a][b])/(n-a+1)
                for j in range(m-1):
                    if (savg[n-a][b][j+1] > upprice and upprice > savg[n-a][b][j]) or (abs(upprice - savg[n-a][b][j]) < 10**-4):
                        uidx = j
                        if upprice == savg[n-a][b][j]:
                            cu = option[a][b][uidx]
                        else:
                            cu = (option[a][b][uidx+1] - option[a][b][uidx])*((upprice-savg[n-a][b][uidx])/(savg[n-a][b][uidx+1]-savg[n-a][b][uidx])) + option[a][b][uidx]
                        break
                lowprice = (savg[n-a-1][b][i] * (n-a) + stock[n-a][b+1])/(n-a+1)
                for k in range(m-1):
                    if (savg[n-a][b+1][k+1] > lowprice and lowprice > savg[n-a][b+1][k]) or (abs(lowprice - savg[n-a][b+1][k+1])< 10**-4):
                        didx = k
                        if lowprice == savg[n-a][b+1][k+1]:
                            cd = option[a][b+1][didx+1]
                        else:
                            cd = (option[a][b+1][didx+1] - option[a][b+1][didx])*((lowprice-savg[n-a][b+1][didx])/(savg[n-a][b+1][didx+1]-savg[n-a][b+1][didx])) + option[a][b+1][didx]
                        break
                if Amer:
                    tmp.append(max(np.exp(-r*t)*(rnp*cu+(1-rnp)*cd),savg[n-a-1][b][i]-K))
                else:
                    tmp.append(np.exp(-r*t)*(rnp*cu+(1-rnp)*cd))
            tmp2.append(tmp)
            tmp = []
        option.append(tmp2)
        tmp2 = []
    return option[-1][0][0]


# Principle of Financial Calculation hw5
def SpreadOption(S1,S2,K,r,q1,q2,sigma1,sigma2,T,n,rho):
    dt = T/n
    x10 = sigma2*np.log(S1) + sigma1*np.log(S2)
    x20 = sigma2*np.log(S1) - sigma1*np.log(S2)
    u1 = (sigma2*(r-q1-(sigma1**2/2))+sigma1*(r-q2-(sigma2**2/2)))*dt + sigma1*sigma2*np.sqrt(2*(1+rho))*np.sqrt(dt)
    d1 = (sigma2*(r-q1-(sigma1**2/2))+sigma1*(r-q2-(sigma2**2/2)))*dt - sigma1*sigma2*np.sqrt(2*(1+rho))*np.sqrt(dt)
    u2 = (sigma2*(r-q1-(sigma1**2/2))-sigma1*(r-q2-(sigma2**2/2)))*dt + sigma1*sigma2*np.sqrt(2*(1-rho))*np.sqrt(dt)
    d2 = (sigma2*(r-q1-(sigma1**2/2))-sigma1*(r-q2-(sigma2**2/2)))*dt - sigma1*sigma2*np.sqrt(2*(1-rho))*np.sqrt(dt)

    x1 = list()
    x2 = list()
    option = list()

    for a in range(n+1):
        x1.append(x10 + (n-a)*u1 + a*d1)
        x2.append(x20 + (n-a)*u2 + a*d2)

    for a in range(n+1):
        tmp = []
        for b in range(n+1):
            s1 = np.exp((x1[a]+x2[b])/(2*sigma2))
            s2 = np.exp((x1[a]-x2[b])/(2*sigma2))
            tmp.append(max(s1-s2-K,0))
        option.append(tmp)

    for i in range(n):
        for a in range(n-i):
            for b in range(n-i):
                option[a][b] = np.exp(-r*dt)*0.25*(option[a][b]+option[a+1][b]+option[a][b+1]+option[a+1][b+1])

    return option[0][0]

# Financial Computation hw3
def MaxRainbowOption(S,K,r,q,sigma,T,rho,flag=0,SimulationNo=10000,RepetitionTime=20):
    average = list()
    count = len(S)
    cov = np.zeros((count,count))

    for i in range(count):
        cov[i][i] = sigma[i]**2 * T
        for j in range(i+1,count):
            cov[i][j] = rho[j+i-1] * sigma[i] * sigma[j] * T
            cov[j][i] = cov[i][j]
#    print cov, "\n"

    cholesky = CholeskyDecomposition(cov,count)
#    print cholesky, "\n"

    for a in range(RepetitionTime):
        option = []
        tmp = []
        # Normal parameter
        for i in range(count):
            # Antithetic variance and moment matching
            if flag != 0:
                tmp2 = np.random.normal(0, 1, SimulationNo//2)
                tmp2 = np.append(tmp2, (-tmp2))
                tmp2 = tmp2/np.std(tmp2)
            else:
                tmp2 = np.random.normal(0, 1, SimulationNo)
            tmp.append(tmp2)
#        print np.cov(tmp), "\n"  # print uncorrelated cov matrix
        if flag == 2:
            cholesky = np.matmul(la.inv(CholeskyDecomposition(np.cov(tmp),count)),cholesky)

        correlatedN = np.transpose(np.matmul(np.transpose(tmp),cholesky))
#        print np.cov(np.vstack(correlatedN)), "\n"  # print correlated matrix
        for i in range(count):
            mean = np.log(S[i])+(r-q[i]-(sigma[i]**2/2.0))*T
            correlatedN[i] = np.exp(correlatedN[i] + mean)

        stock = zip(*correlatedN)

        for i in range(SimulationNo):
            option.append(np.exp(-r*T)*max(max(stock[i])-K,0))

        option = np.array(option)
        average.append(np.average(option))

    mid = np.average(average)
    U = np.average(average) + 2*np.std(average)
    L = np.average(average) - 2*np.std(average)

    return mid, L, U

def LookbackEuroPutMonteCarlo(S,Smax,r,q,sigma,T,n,SimulationNo=10000,RepetitionTime=20):

    option = list()
    dt = T/n

    for a in range(RepetitionTime):
        st = (np.zeros(SimulationNo)+np.log(S)).reshape(SimulationNo,1)
        dw = np.sqrt(dt)*np.random.randn(SimulationNo,n)
        for b in range(n):
            st = np.concatenate((st,st[:,-1].reshape(SimulationNo,1)+(r-q-0.5*sigma**2)*dt+sigma*dw[:,b].reshape(SimulationNo,1)),axis=1)
#            print st
        st = np.exp(st)
#        print st
        path_max = np.maximum(np.apply_along_axis(np.max,1,st),Smax)
        option.append(np.average(np.exp(-r*T)*(path_max - st[:,-1])))

    mid = np.average(option)
    U = mid + 2 * np.std(option)
    L = mid - 2 * np.std(option)

    return mid, L, U

def LookbackPutCRR(S,Smax,r,q,sigma,T,n,Amer=True,bonus=0):
    dt = T/n
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    rnp = (np.exp((r-q)*dt) - d)/(u-d) # Risk Netural Probability

    stock = list()
    maxprice = list()
    option = list()
    tmp = list()
    tmp2 = list()
    sulist = list()

    # Stock price tree
    for a in range(n+1):
        for b in range(a+1):
            tmp.append(S*np.float_power(u,a-b)*np.float_power(d,b))
        stock.append(tmp)
        tmp = []
#    PrintTree(stock)

    # Max price tree
    if bonus == 1:
#        sulist = [max(Smax,S*np.float_power(u,c)) for c in range(n+1)]
        for a in range(n+1):
            for b in range(a+1):
                for c in range(a-b+1):
#                    tmp2 = sulist[:a-b]
                    tmp2.append(max(Smax,S*np.float_power(u,c)))
                tmp.append(tmp2)
                tmp2 = []
            maxprice.append(tmp)
            tmp = []
    else:
        for a in range(n+1):
            for b in range(a+1):
                if a == 0:
                    node_maxprice = [Smax]
                else:
                    if b == 0:
                        node_maxprice = [max(Smax, stock[a][b])]
                    elif b == a:
                        node_maxprice = [Smax]
                    else:
    #                    node_maxprice = maxprice[-1][b-1] gives the pointer of a list to node_maxprice
                        node_maxprice = maxprice[-1][b-1][:]
                        for ele in maxprice[-1][b][:]:
                            if(ele not in node_maxprice):
                                node_maxprice.append(ele)
                        # something wrong??
    #                    if min(node_maxprice) > stock[a][b]:
    #                        node_maxprice.append(stock[a][b])
    #                    node_maxprice.append(stock[a][b])
                tmp.append(node_maxprice)
            maxprice.append(tmp)
            tmp = []

    for a in range(n+1):
        tmp.append([max(ele-stock[-1][a],0) for ele in maxprice[-1][a]])
    option.append(tmp)

    tmp = []
    for a in range(n):
        for b in range(n-a):
            for i in range(len(maxprice[n-a-1][b])):
                upmax = max(stock[n-a][b],maxprice[n-a-1][b][i])
                for j in range(len(maxprice[n-a][b])):
#                    print upmax, maxprice[n-a][b][j]
#                    print upmax == maxprice[n-a][b][j]
                    if abs(upmax - maxprice[n-a][b][j]) < 0.0001:
                        cu = option[a][b][j]
                        break
                lowmax = maxprice[n-a-1][b][i]
                for k in range(len(maxprice[n-a][b+1])):
                    if lowmax == maxprice[n-a][b+1][k]:
                        cd = option[a][b+1][k]
                        break
                if Amer == True:
                    tmp.append(max(np.exp(-r*dt)*(rnp*cu+(1-rnp)*cd),maxprice[n-a-1][b][i]-stock[n-a-1][b]))
                else:
                    tmp.append(np.exp(-r*dt)*(rnp*cu+(1-rnp)*cd))
            tmp2.append(tmp)
            tmp = []
        option.append(tmp2)
        tmp2 = []
#    PrintTree(option)
    
    return option[-1][0][0]

def CheukVorst1997(S,Smax,r,q1,sigma,T,n,Amer=True):
    dt = T/n
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
#    rnp = (np.exp((r-q1)*t) - d)/(u-d) # Risk Netural Probability
    mu = np.exp(r*dt)
#    q = (rnp*u)/(rnp*u+(1-rnp)*d)
    q = 1 - (mu*u-1)/(mu*(u-d)) # CV1997 risk neutral prob.

    tmp = list()
    stock = [[1]]
    option = list()

    # Stock price tree
    for a in range(1,n+1):
        for b in range(a+1):
            tmp.append(np.float_power(u,a-b))
        stock.append(tmp)
        tmp = []

    option.append([max(ele-1,0) for ele in stock[-1][:]])

    for a in range(n):
        tmp = []
        for b in range(n-a):
            if b == (n-a-1):
                tmp.append((q*option[-1][b]+(1-q)*option[-1][b+1]))
            else:
                if Amer:
                    tmp.append(max((q*option[-1][b]+(1-q)*option[-1][b+2]),stock[n-a-1][b]-1))
                else:
                    tmp.append((q*option[-1][b]+(1-q)*option[-1][b+2]))

        option.append(tmp)

    return option[-1][0]*S

# Principle of financial computation hw7
def LeastSquareMethodMonteCarlo(S,K,r,q,sigma,T,n=3,SimulationNo=20000,stage=2):
    def V(X,Y):
        XX = np.matmul(X.T,X)
        XY = np.matmul(X.T,Y.reshape(X.shape[0],1))
        return np.matmul(la.inv(XX),XY)
    
    dt = T/n
    option = list()
    beta = list()

    # Srock price tree
    st = (np.zeros(SimulationNo)+np.log(S)).reshape(SimulationNo,1)
    dw = np.sqrt(dt)*np.random.randn(SimulationNo,n)
    for b in range(n):
        st = np.concatenate((st,st[:,-1].reshape(SimulationNo,1)+(r-q-0.5*sigma**2)*dt+sigma*dw[:,b].reshape(SimulationNo,1)),axis=1)
    st = np.exp(st)

    option = np.maximum(K-st[:,-1],0).reshape(SimulationNo,1)

    for a in range(n-1):
        ITM = list()
        value = list()
        
        # Holding value function form
        x0 = np.ones((SimulationNo,1))
        x = st[:,n-a-1].reshape(SimulationNo,1)
        x2 = np.square(x)
        X = np.concatenate((x0,x,x2),axis=1)
        
        Y = np.exp(-r*dt)*option

        reg = np.concatenate((Y,X),axis=1)

        for ele in reg:
            if K > ele[2]:
                ITM.append(ele)
        ITM = np.array(ITM)

        beta.append(V(ITM[:,1:],ITM[:,0].reshape(ITM.shape[0],1)))

        for i in range(len(X)):
            if X[i][1] > K:
                value.append(Y[i][0])
            else:
                if np.dot(X[i],beta[a]) > (K-X[i][1]):
                    value.append(Y[i][0])
                elif np.dot(X[i],beta[a]) < (K-X[i][1]):
                    value.append(K-X[i][1])
        option = np.array(value).reshape(SimulationNo,1)

    Stage2Price = np.maximum(K-S,np.average(np.exp(-r*dt)*option.flatten()))
    print "Stage2 price:", Stage2Price

    if stage == 3:
        st = (np.zeros(SimulationNo)+np.log(S)).reshape(SimulationNo,1)
        dw = np.sqrt(dt)*np.random.randn(SimulationNo,n)
        for b in range(n):
            st = np.concatenate((st,st[:,-1].reshape(SimulationNo,1)+(r-q-0.5*sigma**2)*dt+sigma*dw[:,b].reshape(SimulationNo,1)),axis=1)
        st = np.exp(st)
    
        option = np.maximum(K-st[:,-1],0).reshape(SimulationNo,1)
        for a in range(n-1):
            ITM = list()
            value = list()
            
            # Holding value function form
            x0 = np.ones((SimulationNo,1))
            x = st[:,n-a-1].reshape(SimulationNo,1)
            x2 = np.square(x)
            X = np.concatenate((x0,x,x2),axis=1)
            
            Y = np.exp(-r*dt)*option

            for i in range(len(X)):
                if X[i][1] > K:
                    value.append(Y[i][0])
                else:
                    if np.dot(X[i],beta[a]) > (K-X[i][1]):
                        value.append(Y[i][0])
                    elif np.dot(X[i],beta[a]) < (K-X[i][1]):
                        value.append(K-X[i][1])
            option = np.array(value).reshape(SimulationNo,1)
            
    option = np.maximum(K-S,np.average(np.exp(-r*dt)*option.flatten()))
    if stage == 3:
        print "Stage3 price:", option

    return beta, option

def IVNewton(S,K,r,q,sigma,T,p,marketprice,sigmaLeft=0,sigmaRight=1,n=0):
    vega = (S*(T**0.5)*norm.pdf(d1(S,K,r,q,sigma,T)))/100
    return sigma - ((p-marketprice)/vega)*0.005, sigmaLeft, sigmaRight

def IVBisection(S,K,r,q,sigma,T,p,marketprice,sigmaLeft=0,sigmaRight=1,n=0):
    if p - marketprice > 0:
        sigmaRight = sigma
        sigmaLeft = sigmaLeft
    else:
        sigmaLeft = sigma
        sigmaRight = sigmaRight
    return (sigmaLeft+sigmaRight)/2, sigmaLeft, sigmaRight

def BSImpliedVolitility(PriceFunction,SearchFunction,S,K,r,q,T,marketprice,error=10**-4):
    sigma = 0.2
    sigmaLeft = 0
    sigmaRight = 1
    p = PriceFunction(S,K,r,q,sigma,T)

    while(abs(p-marketprice) > error):
        tmp = sigma
        sigma, sigmaLeft, sigmaRight = SearchFunction(S,K,r,q,sigma,T,p,marketprice,sigmaLeft,sigmaRight)
        lastSigma = tmp
        print "Simga =", sigma
        p = PriceFunction(S,K,r,q,sigma,T)
    return sigma

def CRRImpliedVolitility(PriceFunction,SearchFunction,S,K,r,q,T,n,marketprice,error=10**-4):
    sigma = 0.2
    sigmaLeft = 0
    sigmaRight = 1
    p = PriceFunction(S,K,r,q,sigma,T,n)

    while(abs(p-marketprice) > error):
        tmp = sigma
        sigma, sigmaLeft, sigmaRight = SearchFunction(S,K,r,q,sigma,T,p,marketprice,sigmaLeft,sigmaRight,n)
        lastSigma = tmp
        print "Simga =", sigma
        p = PriceFunction(S,K,r,q,sigma,T,n)
    return sigma
 
# FC bonus2
def ImplicitFiniteDifference(S,K,r,q,sigma,T,Smin,Smax,m,n,Amer,Put):
    dt = T/n
    W = np.zeros((m-1,m+1))
    option = np.zeros((m+1,n+1))

    stock = np.linspace(Smin,Smax,m+1)
#    print stock
    sindex = list(stock).index(S)
#    print sindex

    if Put:
        for i in range(m+1):
            for j in range(n+1):
                if i == 0:
#                    option[i,j] = K*np.exp(-r*(n-j)*t) - stock[i]*np.exp(-q*(n-j)*t)
                    option[i,j] = K - stock[i]
                elif i == m:
                    option[i,j] = 0
                if j == n:
                    option[i,j] = max(K - stock[i],0)
    else:
        for i in range(m+1):
            for j in range(n+1):
                if i == 0:
                    option[i,j] = 0
                elif i == m:
#                    option[i,j] = stock[i]*np.exp(-q*(n-j)*t) - K*np.exp(-r*(n-j)*t)
                    option[i,j] = stock[i] - K
                if j == n:
                    option[i,j] = max(stock[i]-K,0)
#    print option
    # W
    for j in range(m-1):
        a = (r-q)*0.5*(j+1)*dt-0.5*((sigma*(j+1))**2)*dt
        b = 1 + (((sigma*(j+1))**2)+r)*dt
        c = -(r-q)*0.5*(j+1)*dt-0.5*(sigma*(j+1))**2*dt
        W[j,j] = a
        W[j,j+1] = b 
        W[j,j+2] = c
#    print W
    inputW = np.delete(W,0,1)
    inputW = np.delete(inputW,-1,1)

    for i in range(n):
        Y = copy.copy(option[1:-1,-i-1].reshape(m-1,1))
        Y[0,0] = Y[0,0] - W[0,0]*option[0,-i-2]
        Y[-1,0] = Y[-1,0] - W[-1,-1]*option[-1,-i-2]
        tmp = np.matmul(la.inv(inputW),Y)
        for k in range(m-1):
            if Amer:
                if Put:
                    option[k+1,n-i-1] = max(tmp[k,0],K-stock[k+1])
                else:
                    option[k+1,n-i-1] = max(tmp[k,0],stock[k+1]-K)
            else:
                option[k+1,n-i-1] = tmp[k,0]
#    print option

    return option[sindex,0]

def ExplicitFiniteDifference(S,K,r,q,sigma,T,Smin,Smax,m,n,Amer,Put):
    dt = T/n
    W = np.zeros((m-1,m+1))
    option = np.zeros((m+1,n+1))

    stock = np.linspace(Smin,Smax,m+1)
#    print stock
    sindex = list(stock).index(S)
#    print sindex

    if Put:
        for i in range(m+1):
            for j in range(n+1):
                if i == 0:
#                    option[i,j] = K*np.exp(-r*(n-j)*t) - stock[i]*np.exp(-q*(n-j)*t)
                    option[i,j] = K - stock[i]
                elif i == m:
                    option[i,j] = 0
                if j == n:
                    option[i,j] = max(K - stock[i],0)
    else:
        for i in range(m+1):
            for j in range(n+1):
                if i == 0:
                    option[i,j] = 0
                elif i == m:
#                    option[i,j] = stock[i]*np.exp(-q*(n-j)*t) - K*np.exp(-r*(n-j)*t)
                    option[i,j] = stock[i]-K
                if j == n:
                    option[i,j] = max(stock[i]-K,0)
#    print option
    # W
    for j in range(m-1):
        a = (-0.5*(r-q)*(j+1)*dt+0.5*((sigma*(j+1))**2)*t)/(1+r*dt)
        b = (1-((sigma*(j+1))**2)*t)/(1+r*dt)
        c = (0.5*(r-q)*(j+1)*dt+0.5*((sigma*(j+1))**2)*t)/(1+r*dt)
        W[j,j] = a
        W[j,j+1] = b 
        W[j,j+2] = c
#    print W

    inputW = W

    for i in range(n):
        Y = copy.copy(option[:,-i-1].reshape(m+1,1))
        tmp = np.matmul(inputW,Y)
        for k in range(m-1):
            if Amer:
                if Put:
                    option[k+1,n-i-1] = max(tmp[k,0],K-stock[k+1])
                else:
                    option[k+1,n-i-1] = max(tmp[k,0],stock[k+1]-K)
            else:
                option[k+1,n-i-1] = tmp[k,0]
#    print option
    return option[sindex,0]
