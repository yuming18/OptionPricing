from __future__ import division
import math
from scipy.stats import norm, mvn, multivariate_normal
import numpy as np
import numpy.linalg as la

def RMSE(true, price):
    s = 0
    for i in range(len(price)):
        s = s + (price[i]-true[i])**2
    rmse = s/float(len(price))
    return rmse**0.5

def NormalCDF(x):
    if x > 0:
        pdf = (1/np.sqrt(2*np.pi)) * np.exp(np.square(x)/-2.0)
        k = 1/(1 + 0.2316419 * x)
        cdf = 1-pdf*(0.319381530*k-0.356563782*k*k+1.781477937*k*k*k-1.821255978*k*k*k*k+1.330274429*k*k*k*k*k)
        return cdf
    else:
       return NormalCDF(-x)

def d1(S,K,r,q,sigma,T):
    return (np.log(S/float(K))+(r-q+(sigma**2)/2.0)*T) / (sigma*(T**0.5))

def d2(S,K,r,q,sigma,T):
    return d1(S,K,r,q,sigma,T)-(sigma*(T**0.5))

def BS_EuroCall(S,K,r,q,sigma,T):
    D1 = d1(S,K,r,q,sigma,T)
    D2 = d2(S,K,r,q,sigma,T)
    return S * np.exp(-q*T) * norm.cdf(D1) - K*np.exp(-r*T)*norm.cdf(D2)

def BS_EuroPut(S,K,r,q,sigma,T):
    D1 = d1(S,K,r,q,sigma,T)
    D2 = d2(S,K,r,q,sigma,T)
    return K*np.exp(-r*T)*norm.cdf(-D2) - S * np.exp(-q*T) * norm.cdf(-D1)

def CRR_EuroCall(S,K,r,q,sigma,T,n):
    t = T/float(n)
    u = np.exp(sigma*np.sqrt(t))
    d = 1/u
    rnp = (np.exp(r*t) - d)/(u-d) # Risk Netural Probability
    stock = []
    for a in range(n+1):
        stock.append(S*np.float_power(u,n-a)*np.float_power(d,a))
    option = [x-K if x>K else 0 for x in stock]
    for a in range(n):
        for b in range(len(option)-1):
            option[b] = np.exp(-1*r*t) * (rnp * option[b] + (1-rnp) * option[b+1])
    return option[0]

# Principle of Financial Calculation hw1
def CallOnCall(S,K1,K2,r,q,sigma,t,T):

    def FindS(S,K1,K2,r,q,sigma,T):
        target = S
        AcceptedError = 0.0000001
        premium = BS_EuroCall(target,K2,r,q,sigma,T)
        while(abs(premium - K1) > AcceptedError):
            delta = norm.cdf(d1(target,K2,r,q,sigma,T))
            target = target - 0.1 * (premium-K1)/delta
            premium = BS_EuroCall(target,K2,r,q,sigma,T)
#        print target
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
    
    def FindCriticalPrice(S,K,r,q,sigma,T,AcceptedError=0.0001):
        target = K
        premium = BS_EuroPut(target,K,r,q,sigma,T)

        # K - target is the early exercise value 
        # premium is the holding value
        while(abs((K - target) - premium) > AcceptedError):
            delta = -norm.cdf(-d1(target,K,r,q,sigma,T))
            target = target - ((K-target)-premium)/delta
            premium = BS_EuroPut(target,K,r,q,sigma,T)
        return target


    t1 = T/2.0
    CriticalPrice = FindCriticalPrice(S,K,r,q,sigma,t1)
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

    def Find2TPrice(S,K,r,q,sigma,T,AcceptedError = 0.0001):
        target = K
        premium = BS_EuroPut(target,K,r,q,sigma,T)
#        print premium
        while(abs((K - target) - premium) > AcceptedError):
            premium = BS_EuroPut(target,K,r,q,sigma,T)
            delta = -norm.cdf(-d1(target,K,r,q,sigma,T))
            target = target - 0.1 * ((K - target) - premium)/delta
#        print target
        return target

    t1 = T/3.0
    t2 = (2*T)/3.0

    TPrice = FindTPrice(S,K,r,q,sigma,t2)
    print "Early Exercise Critical Price at T/3 is:", TPrice
    T2Price = Find2TPrice(S,K,r,q,sigma,t1)
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
    t = T/n
    u = np.exp(sigma*np.sqrt(t))
    d = 1/u
    rnp = (np.exp(r*t) - d)/(u-d) # Risk Netural Probability
    leef = []
    option = []
    for a in range(n):
        leef.append(S*np.float_power(u,n-(a+1))*np.float_power(d,a))
    for s in leef:
        option.append(BS_EuroCall(s,K,r,q,sigma,t))
    for a in range(n-1):
        for b in range(len(leef)-1):
            option[b] = np.exp(-1*r*t) * (rnp * option[b] + (1-rnp) * option[b+1])
    return option[0]

def EFB_EuroCall(S,K,r,q,sigma,T,n): #Extrapolated Flexible Binomial
    t = T/n
    u = np.exp(sigma*np.sqrt(t))
    d = 1/u
    ita = int(round((np.log(K/S)-n*np.log(d))/(np.log(u/d))))
    lam = (np.log(K/S)-(2*ita-n)*sigma*np.sqrt(t))/(n*(sigma**2)*t)

    u = np.exp(sigma*np.sqrt(t)+lam*(sigma**2)*t)
    d = np.exp(-sigma*np.sqrt(t)+lam*(sigma**2)*t)
    rnp = (np.exp(r*t) - d)/(u-d)
#    print S*np.float_power(u,ita)*np.float_power(d,n-ita)

    stock = []
    option = []
    for a in range(n+1):
        stock.append(S*np.float_power(u,n-a)*np.float_power(d,a))
    option = [s-K if s > K else 0 for s in stock]
    for a in range(n):
        for b in range(len(option)-1):
            option[b] = np.exp(-1*r*t) * (rnp * option[b] + (1-rnp) * option[b+1])
    return option[0]
    
def GCRRXPC_EuroCall(S,K,r,q,sigma,T,n):
    t = T/n
    a = (np.log(K/S))/(sigma*np.sqrt(n*T))
    lam = a + np.sqrt(a**2+1)

    u = np.exp(lam*sigma*np.sqrt(t))
    d = np.exp((-sigma*np.sqrt(t))/lam)
    rnp = (np.exp(r*t) - d)/(u-d)

    stock = []
    option = []
    for a in range(n+1):
        stock.append(S*np.float_power(u,n-a)*np.float_power(d,a))
    option = [s-K if s > K else 0 for s in stock]
    for a in range(n):
        for b in range(len(option)-1):
            option[b] = np.exp(-1*r*t) * (rnp * option[b] + (1-rnp) * option[b+1])
    return option[0]

def LEIS_EuroCall(S,K,r,q,sigma,T,n):
    t = T/n
    z1 = d1(S,K,r,q,sigma,T)
    z2 = d2(S,K,r,q,sigma,T)
    p1 = 0.5+np.sqrt((0.25-0.25*np.exp(-np.square(z1/(n+(1/3)))*(n+(1/6)))))
    p2 = 0.5+np.sqrt((0.25-0.25*np.exp(-np.square(z2/(n+(1/3)))*(n+(1/6)))))
    u = (np.exp(r*t)*p1)/p2
    d = (np.exp(r*t)-p2*u)/(1-p2)

    rnp = (np.exp(r*t) - d)/(u-d)
    stock = []
    option = []
    for a in range(n+1):
        stock.append(S*np.float_power(u,n-a)*np.float_power(d,a))
    option = [s-K if s > K else 0 for s in stock]
    for a in range(n):
        for b in range(len(option)-1):
            option[b] = np.exp(-1*r*t) * (rnp * option[b] + (1-rnp) * option[b+1])
    return option[0]
    

def HZ_EuroCall(S,K,r,q,sigma,T,n):
    
    def integrate(K,a,b,N):
        x = np.linspace(a+(b-a)/(2*N), b-(b-a)/(2*N), N)
#        x = np.linspace(a, b, N)
        y = list()
        for ele in x:
            y.append(max(ele-K,0.0))
        y = np.array(y)
        area = np.sum(y)*(b-a)/N
        return area

    t = T/n

    lam = 1.1
    u = np.exp(lam*sigma*np.sqrt(t))
    d = np.exp((-sigma*np.sqrt(t))/lam)
    rnp = (np.exp(r*t) - d)/(u-d)

    dx = S*0.5*sigma*np.sqrt(t)*(lam+(1/lam))-S

    stock = []
    option = []

    for a in range(n+1):
        stock.append(S*np.float_power(u,n-a)*np.float_power(d,a))
    option = [integrate(K,s-dx,s+dx,100)/(2*dx) for s in stock]
    for a in range(n):
        for b in range(len(option)-1):
            option[b] = np.exp(-1*r*t) * (rnp * option[b] + (1-rnp) * option[b+1])
    return option[0]
