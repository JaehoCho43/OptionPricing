import numpy as np
import scipy
from scipy.sparse import diags
from scipy.sparse.linalg import splu
import analytics.analytics_utils as analytics_utils
from typing import override


class EuropeanContract:
    def __init__(self, rd, rf, T, vol):
        self.rd = rd
        self.rf = rf
        self.T = T
        self.vol = vol

    def intrinsicValue(self, spots):
        return np.zeros_like(spots)
    
    def priceMonteCarlo(self, spots, numPaths):
        np.random.seed(42)
        spotSize = spots.size
        sims = np.exp((self.rd - self.rf - 0.5 * self.vol ** 2) * self.T + self.vol * np.sqrt(self.T) * np.random.randn(spotSize, numPaths))
        spotSims = np.array([spots]).T * sims
        
        return self.df * self.intrinsicValue(spotSims).mean(axis = 1)
    
    def pricePDEwithSpots(self, spotL, spotU, numSpots, numT, spotMultiplicativeOffset = 0.5, theta = 0.5, numRanaccher = 0, getAllResults = False):
        logSpot_mesh = np.linspace(np.log(spotL * (1 - spotMultiplicativeOffset)), np.log(spotU * (1 + spotMultiplicativeOffset)), numSpots)
        hy = logSpot_mesh[1] - logSpot_mesh[0]
        ht = self.T / numT

        result = np.zeros((numT, numSpots))
        result[0,1:-1] = self.intrinsicValue(np.exp(logSpot_mesh))[1:-1]

        A = ht * (self.rd - self.rf - 0.5 * self.vol ** 2) / (2 * hy) * diags(diagonals = [-1,0,1], offsets = [-1,0,1], shape = (numSpots, numSpots))
        B = ht * self.vol ** 2 / (2 * hy ** 2) * diags(diagonals = [1,-2,1], offsets = [-1,0,1], shape = (numSpots, numSpots))
        I = diags([1],[0], shape = (numSpots, numSpots))

        Ml = (1 + ht * self.rd * theta) * I - theta * (A + B)
        Mr = (1 - ht * self.rd * (1 - theta)) * I + (1 - theta) * (A + B)
        MlThetaOne = (1 + ht * self.rd) * I - (A + B)
        
        for i in range(1, numRanaccher + 1):
            result[i] = splu(MlThetaOne.tocsc()).solve(result[i-1])

        for i in range(numRanaccher + 1, numT):
            b = Mr @ result[i-1]
            result[i] = splu(Ml.tocsc()).solve(b)

        X_PDE = np.exp(logSpot_mesh)
        result = result[:,(spotL <= X_PDE) & (X_PDE <= spotU)]
        X_PDE = X_PDE[(spotL <= X_PDE) & (X_PDE <= spotU)]

        return (X_PDE, result) if getAllResults else (X_PDE, result[-1])

    
class EuroVanilla(EuropeanContract):
    def __init__(self, rd, rf, T, vol, K, cp, pt = analytics_utils.PaymentType.CASHDOM):
        super().__init__(rd, rf, T, vol)
        self.K = K
        self.cp = cp
        self.w = 1 if (self.cp == analytics_utils.CallPut.CALL) else -1
        self.df = np.exp(-self.rd * self.T) if (pt == analytics_utils.PaymentType.CASHDOM) else np.exp(-self.rf * self.T)
        

    @override
    def intrinsicValue(self, spots):
        arr = self.w * (spots - self.K)
        return np.where(arr > 0, arr, 0)

    def priceExact(self, S):
        forward = S * np.exp((self.rd - self.rf) * self.T)
        d1 = (np.log(forward / self.K) + 0.5 * (self.vol ** 2) * self.T)/(self.vol * np.sqrt(self.T))
        d2 = d1 - self.vol * np.sqrt(self.T)
        return self.df * self.w * (forward * scipy.stats.norm.cdf(self.w * d1) - self.K * scipy.stats.norm.cdf(self.w * d2))
    
class Binary(EuropeanContract):
    def __init__(self, rd, rf, T, vol, K, cp, pt = analytics_utils.PaymentType.CASHDOM):
        super().__init__(rd, rf, T, vol)
        self.K = K
        self.cp = cp
        self.w = 1 if (self.cp == analytics_utils.CallPut.CALL) else -1
        self.df = np.exp(-self.rd * self.T) if (pt == analytics_utils.PaymentType.CASHDOM) else np.exp(-self.rf * self.T)
        
    @override
    def intrinsicValue(self, spots):
        arr = self.w * (spots - self.K)
        return np.where(arr > 0, 1, 0)

    def priceExact(self, S):
        forward = S * np.exp((self.rd - self.rf) * self.T)
        d1 = (np.log(forward / self.K) + 0.5 * (self.vol ** 2) * self.T)/(self.vol * np.sqrt(self.T))
        d2 = d1 - self.vol * np.sqrt(self.T)
        return self.df * scipy.stats.norm.cdf(self.w * d2)

class EuroKIKO(EuropeanContract):
    def __init__(self, rd, rf, T, vol, K, cp, B, bt, pt = analytics_utils.PaymentType.CASHDOM):
        super().__init__(rd, rf, T, vol)
        self.K = K
        self.cp = cp
        self.B = B
        self.bt = bt
        self.w = 1 if (self.cp == analytics_utils.CallPut.CALL) else -1
        self.df = np.exp(-self.rd * self.T) if (pt == analytics_utils.PaymentType.CASHDOM) else np.exp(-self.rf * self.T)
        
    @override
    def intrinsicValue(self, spots):
        arr = self.w * (spots - self.K)
        match (self.cp, self.bt):
            case (analytics_utils.CallPut.CALL, analytics_utils.BarrierType.KO):
                return np.where(arr > 0, arr, 0) * np.where(spots > self.B, 0, 1)
            case (analytics_utils.CallPut.PUT, analytics_utils.BarrierType.KI):
                return np.where(arr > 0, arr, 0) * np.where(spots > self.B, 0, 1)
            case _:
                return np.where(arr > 0, arr, 0) * np.where(spots < self.B, 0, 1)
            
class BinaryKO(EuropeanContract):
    def __init__(self, rd, rf, T, vol, K, cp, B, pt = analytics_utils.PaymentType.CASHDOM):
        super().__init__(rd, rf, T, vol)
        self.K = K
        self.cp = cp
        self.B = B
        self.w = 1 if (self.cp == analytics_utils.CallPut.CALL) else -1
        self.df = np.exp(-self.rd * self.T) if (pt == analytics_utils.PaymentType.CASHDOM) else np.exp(-self.rf * self.T)
        
    @override
    def intrinsicValue(self, spots):
        arr = self.w * (spots - self.K)
        arrB = self.w * (spots - self.B)
        return np.where((arr > 0) & (arrB < 0), 1, 0)


        



