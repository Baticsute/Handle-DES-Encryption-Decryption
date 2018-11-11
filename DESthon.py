import  numpy as np
from random import randint
## Key 64 bit : 8 rows and 8 columns
## Each Character of plaintex is 8 bit ( 256 ASCII )

IP = [58, 50, 42, 34, 26, 18, 10, 2,
      60, 52, 44, 36, 28, 20, 12, 4,
      62, 54, 46, 38, 30, 22, 14, 6,
      64, 56, 48, 40, 32, 24, 16, 8,
      57, 49, 41, 33, 25, 17, 9, 1,
      59, 51, 43, 35, 27, 19, 11, 3,
      61, 53, 45, 37, 29, 21, 13, 5,
      63, 55, 47, 39, 31, 23, 15, 7]

## 56 bits , Matrix 7x8
PC_1 = [57, 49, 41, 33, 25, 17, 9,
        1, 58, 50, 42, 34, 26, 18,
        10, 2, 59, 51, 43, 35, 27,
        19, 11, 3, 60, 52, 44, 36,
        63, 55, 47, 39, 31, 23, 15,
        7, 62, 54, 46, 38, 30, 22,
        14, 6, 61, 53, 45, 37, 29,
        21, 13, 5, 28, 20, 12, 4]

PC_2 = [14  ,  17 ,  11   , 24  ,   1   , 5,
                  3  ,  28  , 15  ,   6  ,  21,   10,
                 23  ,  19  , 12  ,   4   , 26 ,   8,
                 16   ,  7 ,  27 ,   20 ,   13   , 2,
                 41   , 52  , 31   , 37   , 47 ,  55,
                 30  ,  40 ,  51 ,   45  ,  33  , 48,
                 44  ,  49  , 39  ,  56  ,  34 ,  53,
                 46   , 42  , 50  ,  36  ,  29  , 32 ]

E = [32, 1, 2, 3, 4, 5,
     4, 5, 6, 7, 8, 9,
     8, 9, 10, 11, 12, 13,
     12, 13, 14, 15, 16, 17,
     16, 17, 18, 19, 20, 21,
     20, 21, 22, 23, 24, 25,
     24, 25, 26, 27, 28, 29,
     28, 29, 30, 31, 32, 1]

P = [16, 7, 20, 21, 29, 12, 28, 17,
     1, 15, 23, 26, 5, 18, 31, 10,
     2, 8, 24, 14, 32, 27, 3, 9,
     19, 13, 30, 6, 22, 11, 4, 25]

SHIFT = [1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1]

IP_1 = [40, 8, 48, 16, 56, 24, 64, 32,
        39, 7, 47, 15, 55, 23, 63, 31,
        38, 6, 46, 14, 54, 22, 62, 30,
        37, 5, 45, 13, 53, 21, 61, 29,
        36, 4, 44, 12, 52, 20, 60, 28,
        35, 3, 43, 11, 51, 19, 59, 27,
        34, 2, 42, 10, 50, 18, 58, 26,
        33, 1, 41, 9, 49, 17, 57, 25]

S_BOX = [

    [[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
     [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
     [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
     [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],
     ],

    [[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
     [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
     [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
     [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],
     ],

    [[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
     [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
     [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
     [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],
     ],

    [[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
     [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
     [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
     [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],
     ],

    [[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
     [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
     [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
     [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],
     ],

    [[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
     [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
     [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
     [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],
     ],

    [[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
     [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
     [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
     [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],
     ],

    [[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
     [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
     [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
     [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
     ]
]

## 64bits : 8x8
Key = [0,0,0,1,0,0,1,1,0,0,1,1,0,1,0,0,0,1,0,1,0,1,1,1,0,1,1,1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,1,1,1,0,0,1,1,0,1,1,1,1,1,1,1,1,1,0,0,0,1]

M = [0,0,0,0,0,0,0,1,0,0,1,0 ,0,0,1,1,0,1,0,0,0,1,0,1,0,1,1,0 ,0,1,1,1,1,0,0,0,1,0,0,1,1,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1 ,1,1,1,0,1,1,1,1]

def Permutation_PC1(Key): ## input is a 64 bits Key

    output = []

    for i in PC_1:
        output.append(Key[i-1])
    C = output[0:28]  ## C = C0
    D = output[28:]   ## D = D0
    ## Output is a 56 bits Key : 7x8 , SubKeys 24 bits (4x7) C and D
    return output , C , D

def LeftShift_CD(C0_D0_Keys):
    CD_Keys = []
    C0 = C0_D0_Keys[0:28]
    D0 = C0_D0_Keys[28:]
    CD_Keys.append(C0_D0_Keys)
    for i in range(len(SHIFT)):
       C0 = C0[SHIFT[i]:] + C0[0:SHIFT[i]]
       D0 = D0[SHIFT[i]:] + D0[0:SHIFT[i]]
       CD_TEMP = C0 + D0
       CD_Keys.append(CD_TEMP)
    return  CD_Keys
def Permutation_PC2(CDn_Key): #Convert from 56bits to 48bits
    Kn = []
    for i in PC_2:
        Kn.append(CDn_Key[i-1])
    return  Kn
def Create_16_Sub_Keys(CD_Keys): ## Create 16 Sub Keys K , 1 <= n <= 16
    K_Keys = []  ## An Array with 17 element , K_Keys[0] is unuse with 32 zeros
    K_Keys.append([0] * 32)

    for i in range(1,17):
        K_Keys.append(Permutation_PC2(CD_Keys[i]))

    return K_Keys ## Each Key is 48bits lenght

def Permutation_IP(Binary_Plaintext): ## 64 bits to 64 bits

    output = []
    for i in IP:
        output.append(Binary_Plaintext[i-1])
    L0 = output[0:32]
    R0 = output[32:]
    return output , L0 , R0

def xor_(a,b):
    if(a==b):
        return 0
    else:
        return 1
def XOR(a,b):
    output = []
    for i in range(len(a)):
        output.append(xor_(a[i],b[i]))
    return output


def ReShape_56bits(input): ## Check the right answer by display .

    output = np.reshape(input,(8,7))
    return output

def Display_CD_Keys(CD_Keys):

    for i in range(len(CD_Keys)):
        print("C:",i,CD_Keys[i][0:28])
        print("D:",i,CD_Keys[i][28:])

def E_bits_selection(Rn):

    output = []
    for i in E:
        output.append(Rn[i-1])

    return output
def Permutation_P(F):
    output = []
    for i in P:
        output.append(F[i-1])

    return output

def F_function(Kn,Rn):
    ##print(len(Rn))
    Xor = XOR(Kn,E_bits_selection(Rn)) ## 48bits lenght

    F_value = []
    Xor = np.reshape(Xor,(8,6))
    for i in range(0,8):
        brow = list(Xor[i][0:1]) + list(Xor[i][5:])
        bcolumn = list(Xor[i][1:5])
        row = bit_array_to_dec(brow)
        column = bit_array_to_dec(bcolumn)
        S_box_value = S_BOX[i][row][column]
        S_box_value = dec_to_bit_array(S_box_value)
        F_value.append(S_box_value)
    F_value = list(np.ndarray.flatten(np.array(F_value))) ## 32bits lengh
    F_value = Permutation_P(F_value) ## Final F_Value at Kn and Rn-1
    return F_value
def Permutation_IP_Inverse(LR16):
    output = []
    for i in IP_1:
        output.append(LR16[i-1])
    return output
def SixTeen_Round_Encrypt(Plaintext,K_Keys):
    IP_Rs = Permutation_IP(Plaintext)[0]
    L0 = IP_Rs[0:32]
    R0 = IP_Rs[32:]
    List_LR =  []
    List_LR.append(L0+R0)

    for i in range(1,17):
        Ln = List_LR[i-1][32:] ## Ln = Rn-1
        Rn = XOR(List_LR[i-1][0:32],F_function(K_Keys[i],List_LR[i-1][32:]))
        LRn = list(Ln) + list(Rn)

        List_LR.append(LRn)
    R16= List_LR[16][32:]
    L16= List_LR[16][0:32]
    LR16 = R16 + L16

    Encrypted_Block = Permutation_IP_Inverse(LR16)
    return Encrypted_Block

def SixTeen_Round_Decrypt(Ciphertext,K_Keys):
    ##print(len(Ciphertext))
    IP_Rs = Permutation_IP(Ciphertext)[0]
    L16 = IP_Rs[0:32]
    R16 = IP_Rs[32:]
    List_LR = []
    List_LR.append(L16+R16)
    for i in range(1,17):
        Ln = List_LR[i-1][32:] ## Ln = Rn-1
        Rn = XOR(List_LR[i-1][0:32],F_function(K_Keys[17-i],List_LR[i-1][32:]))
        LRn = list(Ln) + list(Rn)
        List_LR.append(LRn)
    R0= List_LR[16][32:]
    L0= List_LR[16][0:32]
    LR0 = R0 + L0

    Encrypted_Block = Permutation_IP_Inverse(LR0)
    return Encrypted_Block
def bit_array_to_dec(bit_arr):
    bit_arr.reverse()
    total = 0
    for i in range(len(bit_arr)):
        if(bit_arr[i]==1):
            total = total + 2**i
    return  total
def dec_to_bit_array(n,bit_size=4):
    str_bit = bin(n)
    str_bit = str_bit[2:]
    bit_arr = []
    while(len(str_bit)!= bit_size):
        str_bit = "0"+str_bit
    for i in str_bit :
        bit_arr.append(int(i))
    return bit_arr

def Generate_Key_64():

    char_key = ""
    for i in range(8):
        char_key =  char_key + chr(randint(33,122))
    return  char_key

C = [1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]

def chartobin(char,size=8):
    ord_ascii = ord(char)
    arr = bin(ord_ascii)[2:]
    bit_arr = []
    while(len(arr) < size):
        arr = "0" + arr
    for i in arr :
        bit_arr.append(int(i))
    return bit_arr
def stringtobitarr(text):
    arr = list()
    for char in text:
        binval = chartobin(char)
        arr.extend([int(x) for x in list(binval)])
    return arr

def DES_Encrypt(Plaintext,Key):
    Plaintext = PaddString(Plaintext)
    Ciphertex = str()
    for block in range(int(len(Plaintext)/8)):
        plaintext = Plaintext[block*8:(block+1)*8]
        Key_bit = stringtobitarr(Key)
        M = stringtobitarr(plaintext)

        a = Permutation_PC1(Key_bit)[0]
        b = LeftShift_CD(a)
        c = Create_16_Sub_Keys(b)
        d = SixTeen_Round_Encrypt(M,c)

        output = np.reshape(d,(8, 8))

        for i in output:
            number_ascii = bit_array_to_dec(list(i))
            ##print(number_ascii)
            char = chr(number_ascii)
            Ciphertex = Ciphertex + char

    return Ciphertex

def DES_Decrypt(Ciphertext,Key):

    Plaintext = str()
    for block in range(int(len(Ciphertext)/8)):
        ciphertext = Ciphertext[block*8:(block+1)*8]
        Key_bit = stringtobitarr(Key)
        C = stringtobitarr(ciphertext)

        a = Permutation_PC1(Key_bit)[0]
        b = LeftShift_CD(a)
        c = Create_16_Sub_Keys(b)
        d = SixTeen_Round_Decrypt(C,c)

        output = np.reshape(d,(8, 8))

        for i in output:
            number_ascii = bit_array_to_dec(list(i))
            ##print(number_ascii)
            char = chr(number_ascii)
            Plaintext =  Plaintext + char
    count_pad = Plaintext.count('}')
    return Plaintext[:len(Plaintext)-count_pad]
def PaddString(str):
    while(len(str)%8 != 0):
        str = str  + '}'
    return str




