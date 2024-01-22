from enum import Enum

class CallPut(Enum):
    CALL = 1
    PUT = 2

class BarrierType(Enum):
    KO = 1
    KI = 2

class PaymentType(Enum):
    CASHDOM = 1
    CASHFOR = 2
