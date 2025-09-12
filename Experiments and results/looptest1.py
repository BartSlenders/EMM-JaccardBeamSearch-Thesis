import pandas as pd
import statsmodels.api as SM
from EMM_fixed import EMM
from Jaccard import Jaccard_EMM
from pattern_team import pattern_team_DTA, mean_error, pattern_team_weighted_DTA
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    # f = open('results_DTA_pattern_team.csv', 'w')
    # f.write('dataset, w,d,beam search method, MSE, Mean error\n')
    # f.close()
    dmax = 6
    for dataset in ['german-credit-scoring']:
        df = pd.read_csv(f'example/data/{dataset}.csv', sep=";")
        if dataset == 'german-credit-scoring':
            target_columns = ['Duration in months', 'Credit amount']
        elif dataset == 'Housing':
            target_columns = ['lotsize', 'price']
        elif dataset == 'titanic':
            target_columns = ['Fare', 'Survived']

        train_data, test_data = train_test_split(df, test_size=0.1)
        test_data = test_data.reset_index()
        test_data = test_data.drop(['index'], axis=1)
        for w in [#5, 10, 20, 40, 70, 100, 150, 200,
                  250, 300]:
            for method in ['Normal Beam Search', 'Jaccard Beam Search']:
                if method == 'Normal Beam Search':
                    Beam = EMM(width=w, depth=1, evaluation_metric='regression', n_jobs=-1, log_level=21)
                if method == 'Jaccard Beam Search':
                    Beam = Jaccard_EMM(width=w, depth=d, evaluation_metric='regression', n_jobs=-1, log_level=21)
                Beam.set_data(train_data, target_cols=target_columns)
                for d in range(dmax):
                    print(dataset+': '+method+'- '+str(w)+', '+str(d+1))
                    Beam.increase_depth(1)
                    predictions = pattern_team_DTA(Beam.beam.subgroups, test_data, train_data, target_columns, SM.OLS)

                    mse = MSE(test_data[target_columns[-1]], predictions)
                    ME = mean_error(test_data[target_columns[-1]], predictions)
                    f = open('results_DTA_pattern_team.csv', 'a')
                    f.write(f'{dataset},{w},{d+1},{str(method)},{str(mse)},{ME}\n')
                    f.close()

                    weight_predictions = pattern_team_weighted_DTA(Beam.beam.subgroups, test_data, train_data,
                                                                   target_columns, SM.OLS)
                    mse = MSE(test_data[target_columns[-1]], weight_predictions)
                    ME = mean_error(test_data[target_columns[-1]], weight_predictions)
                    f = open('results_DTA_pattern_team.csv', 'a')
                    f.write(f'{dataset},{w},{d+1},{str(method)}-weighted,{str(mse)},{ME}\n')
                    f.close()
