# for data generation
import numpy as np
import pandas as pd


def generate(datasize = 2000, randomvariables = 20, mean = 10, errorsd = 0.5, exceptionality= 1, pollutionset=4):
    result = []

    predictor = list(np.random.normal(mean,3,datasize))         # STANDARD DEVIATION IS LOCKED AT 3 HERE

    variables = [list(np.random.binomial(1,0.4,datasize)) for _ in range(randomvariables)]

    # generate result;
    # result data where first two variables are both 1 is different
    for i in range(datasize):
        v = 10
        if variables[2][i] == 0:
            v+=pollutionset
        elif variables[0][i] == 1 and variables[1][i] == 1:
            v-=exceptionality
        # elif variables[0][i] == 1 and variables[2][i] == 1:
        #     result.append(4 * predictor[i]   + np.random.normal(0,errorsd) )
        result.append((v)* predictor[i]  + np.random.normal(0,errorsd) )


    # create a dataframe with number i as column title with the before generated columns
    df = pd.DataFrame({i:ls for i,ls in enumerate(variables)})

    df['result'] = result
    df['predictor'] = predictor
    return df

def generatemoresets(datasize = 2000, randomvariables = 20, mean = 10, errorsd = 0.5, exceptionality= 1, pollutionset=4, superexceptional=2 ):
    result = []
    predictor = list(np.random.normal(mean,3,datasize))         # STANDARD DEVIATION IS LOCKED AT 3 HERE
    variables = [list(np.random.binomial(1,0.4,datasize)) for _ in range(randomvariables)]
    for i in range(5):
        variables.append(list(np.random.binomial(1,0.1,datasize))) #generate 5 sets die specifiek veel pollution kunnen veroorzaken

    # generate result;
    # result data where first two variables are both 1 is different
    for i in range(datasize):
        v = 10
        if variables[2][i] == 0:
            v+=pollutionset
        if variables[3][i] == 0:
            v-=pollutionset
        if variables[0][i] == 1 and variables[1][i] == 1:
            v-=exceptionality
        if variables[4][i] == 1 and variables[5][i] == 1:
            v+=exceptionality
        if variables[3][i] == 1 and variables[7][i] == 1 and variables[8][i] == 1: #reverse of second pollution set
            v+=superexceptional
        if variables[2][i] == 0 and variables[6][i] == 1 and variables[9][i] == 1: #include first pollution set
            v-=superexceptional
        result.append((v)* predictor[i]  + np.random.normal(0,errorsd) )


    # create a dataframe with number i as column title with the before generated columns
    df = pd.DataFrame({i:ls for i,ls in enumerate(variables)})

    df['result'] = result
    df['predictor'] = predictor
    return df




def generatedepth3(datasize = 2000, randomvariables = 20, mean = 10, errorsd = 0.5, exceptionality= 1, pollutionset=4):
    result = []
    predictor = list(np.random.normal(mean,3,datasize))         # STANDARD DEVIATION IS LOCKED AT 3 HERE
    variables = [list(np.random.binomial(1,0.4,datasize)) for _ in range(randomvariables)]
    # generate result;
    # result data where first two variables are both 1 is different
    for i in range(datasize):
        v = 10
        if variables[2][i] == 0:
            v+=pollutionset
        elif variables[0][i] == 1 and variables[1][i] == 1 and variables[3][i] == 1:
            v-=exceptionality
        result.append((v)* predictor[i]  + np.random.normal(0,errorsd) )


    # create a dataframe with number i as column title with the before generated columns
    df = pd.DataFrame({i:ls for i,ls in enumerate(variables)})

    df['result'] = result
    df['predictor'] = predictor
    return df