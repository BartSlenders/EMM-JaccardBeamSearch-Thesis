import Jaccard as J
from EMM_fixed import EMM
from EMM import EMM as oldEMM
from Generation import generatemoresets#, generate
from copy import deepcopy

#for saving the file
import pandas as pd

resultdict = {'w': [], 'errormargin':[], 'pollutionplus':[], 'exceptionalplus':[], 'superexceptionalplus':[], 'typebeam':[], 'pollution1':[], 'pollution2':[], 'exceptional1':[], 'exceptional2':[], 'exceptionalexclude':[], 'exceptionaloverlap':[]}
iterate = 3


target_columns = ['predictor','result']
for w in [10,25,60]:
    for margin in [0.5, 2, 10]: #standard deviation
        for pollution in [4, 8, 20]:
            for exceptional in [1, 3, 5]:
                for superexceptional in [4, 12, 25]:
                    print(f'NORMAL: This run we test with SD of {margin} and pollution set varying by {pollution} and exceptional set varying by {exceptional}')
                    df = generatemoresets(datasize=2000, randomvariables=20, mean=10, errorsd=margin, pollutionset=pollution, exceptionality=exceptional, superexceptional=superexceptional)
                    Beam = EMM(width=w) 
                    Beam.set_data(deepcopy(df), target_columns)
                    Beam.increase_depth(iterations=iterate)
                    pollution1, pollution2, exceptional1, exceptional2, exceptionalexclude, exceptionaloverlap = False, False, False, False, False, False
                    resultdict['errormargin'].append(margin)            # append relevant information
                    resultdict['pollutionplus'].append(pollution) 
                    resultdict['exceptionalplus'].append(exceptional)
                    resultdict['superexceptionalplus'].append(superexceptional)
                    resultdict['typebeam'].append('normal') 
                    resultdict['w'].append(w)
                    for i, sg in enumerate(Beam.beam.calculate_q()):    # search in q for the subgroup
                        if sg.description.description =={2: 0} and pollution1 == False:
                            resultdict['pollution1'].append( i)
                            pollution1=True
                        if sg.description.description =={3: 0} and pollution2 == False:
                            resultdict['pollution2'].append( i)
                            pollution2=True
                        if sg.description.description =={0:1, 1:1} and exceptional1 == False:
                            resultdict['exceptional1'].append( i)
                            exceptional1=True
                        if sg.description.description =={4:1, 5:1} and exceptional2 == False:
                            resultdict['exceptional2'].append( i)
                            exceptional2=True
                        if sg.description.description == {3:1, 7:1, 8:1} and exceptionalexclude == False:
                            resultdict['exceptionalexclude'].append( i)
                            exceptionalexclude=True
                        if sg.description.description =={2:0, 6:1, 9:1} and exceptionaloverlap == False:
                            resultdict['exceptionaloverlap'].append( i)
                            exceptionaloverlap=True
                    if pollution1==False:
                        resultdict['pollution1'].append(9999)                # append 9999 if we can't find the subgroup
                    if pollution2==False:
                        resultdict['pollution2'].append(9999)                # append 9999 if we can't find the subgroup
                    if exceptional1==False:
                        resultdict['exceptional1'].append(9999)                # append 9999 if we can't find the subgroup
                    if exceptional2==False:
                        resultdict['exceptional2'].append(9999)                # append 9999 if we can't find the subgroup
                    if exceptionalexclude==False:
                        resultdict['exceptionalexclude'].append(9999)                # append 9999 if we can't find the subgroup
                    if exceptionaloverlap==False:
                        resultdict['exceptionaloverlap'].append(9999)                # append 9999 if we can't find the subgroup
                    
                    print(f'JACCARD: This run we test with SD of {margin} and pollution set varying by {pollution} and exceptional set varying by {exceptional}')
                    JBeam = J.Jaccard_EMM(width=w)
                    JBeam.set_data(deepcopy(df), target_columns)
                    JBeam.increase_depth(iterations=iterate)
                    pollution1, pollution2, exceptional1, exceptional2, exceptionalexclude, exceptionaloverlap = False, False, False, False, False, False
                    resultdict['errormargin'].append(margin)            # append relevant information
                    resultdict['pollutionplus'].append(pollution) 
                    resultdict['exceptionalplus'].append(exceptional)
                    resultdict['superexceptionalplus'].append(superexceptional)
                    resultdict['typebeam'].append('Jaccard') 
                    resultdict['w'].append(w)
                    for i, sg in enumerate(JBeam.beam.calculate_q()):   # search in q for the subgroup
                        if sg.description.description =={2: 0} and pollution1 == False:
                            resultdict['pollution1'].append( i)
                            pollution1=True
                        if sg.description.description =={3: 0} and pollution2 == False:
                            resultdict['pollution2'].append( i)
                            pollution2=True
                        if sg.description.description =={0:1, 1:1} and exceptional1 == False:
                            resultdict['exceptional1'].append( i)
                            exceptional1=True
                        if sg.description.description =={4:1, 5:1} and exceptional2 == False:
                            resultdict['exceptional2'].append( i)
                            exceptional2=True
                        if sg.description.description == {3:1, 7:1, 8:1} and exceptionalexclude == False:
                            resultdict['exceptionalexclude'].append( i)
                            exceptionalexclude=True
                        if sg.description.description =={2:0, 6:1, 9:1} and exceptionaloverlap == False:
                            resultdict['exceptionaloverlap'].append( i)
                            exceptionaloverlap=True
                    if pollution1==False:
                        resultdict['pollution1'].append(9999)                # append 9999 if we can't find the subgroup
                    if pollution2==False:
                        resultdict['pollution2'].append(9999)                # append 9999 if we can't find the subgroup
                    if exceptional1==False:
                        resultdict['exceptional1'].append(9999)                # append 9999 if we can't find the subgroup
                    if exceptional2==False:
                        resultdict['exceptional2'].append(9999)                # append 9999 if we can't find the subgroup
                    if exceptionalexclude==False:
                        resultdict['exceptionalexclude'].append(9999)                # append 9999 if we can't find the subgroup
                    if exceptionaloverlap==False:
                        resultdict['exceptionaloverlap'].append(9999)                # append 9999 if we can't find the subgroup
                    
                    # print(f'OLD: This run we test with SD of {margin} and pollution set varying by {pollution} and exceptional set varying by {exceptional}')
                    # oldBeam = oldEMM(width=w, depth=iterate, evaluation_metric='regression')
                    # oldBeam.search(deepcopy(df),  target_columns)
                    # pollution1, pollution2, exceptional1, exceptional2, exceptionalexclude, exceptionaloverlap = False, False, False, False, False, False
                    # resultdict['errormargin'].append(margin)            # append relevant information
                    # resultdict['pollutionplus'].append(pollution) 
                    # resultdict['exceptionalplus'].append(exceptional)
                    # resultdict['superexceptionalplus'].append(superexceptional)
                    # resultdict['typebeam'].append('Old') 
                    # resultdict['w'].append(w)
                    # for i, sg in enumerate(oldBeam.beam.subgroups):     # oldEMM was implemented poorly, so always stores q in subgroups
                    #     if sg.description.description =={2: 0} and pollution1 == False:
                    #         resultdict['pollution1'].append( i)
                    #         pollution1=True
                    #     if sg.description.description =={3: 0} and pollution2 == False:
                    #         resultdict['pollution2'].append( i)
                    #         pollution2=True
                    #     if sg.description.description =={0:1, 1:1} and exceptional1 == False:
                    #         resultdict['exceptional1'].append( i)
                    #         exceptional1=True
                    #     if sg.description.description =={4:1, 5:1} and exceptional2 == False:
                    #         resultdict['exceptional2'].append( i)
                    #         exceptional2=True
                    #     if sg.description.description == {3:1, 7:1, 8:1} and exceptionalexclude == False:
                    #         resultdict['exceptionalexclude'].append( i)
                    #         exceptionalexclude=True
                    #     if sg.description.description =={2:0, 6:1, 9:1} and exceptionaloverlap == False:
                    #         resultdict['exceptionaloverlap'].append( i)
                    #         exceptionaloverlap=True
                    # if pollution1==False:
                    #     resultdict['pollution1'].append(9999)                # append 9999 if we can't find the subgroup
                    # if pollution2==False:
                    #     resultdict['pollution2'].append(9999)                # append 9999 if we can't find the subgroup
                    # if exceptional1==False:
                    #     resultdict['exceptional1'].append(9999)                # append 9999 if we can't find the subgroup
                    # if exceptional2==False:
                    #     resultdict['exceptional2'].append(9999)                # append 9999 if we can't find the subgroup
                    # if exceptionalexclude==False:
                    #     resultdict['exceptionalexclude'].append(9999)                # append 9999 if we can't find the subgroup
                    # if exceptionaloverlap==False:
                    #     resultdict['exceptionaloverlap'].append(9999)                # append 9999 if we can't find the subgroup

df = pd.DataFrame(resultdict)
df.to_csv('SecondExcperiment.csv', index=False)