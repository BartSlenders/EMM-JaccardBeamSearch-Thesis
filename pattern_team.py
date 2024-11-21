from numpy import mean


def pattern_team_DTA(subgroups, testdata, traindata, target, model_method):

    if hasattr(model_method, '__call__'):
        pass
    else:
        raise ValueError(f"this is not a callable model: {model_method}")

    predictions = [[] for _ in testdata.index]
    # make models and save the predictions for each model
    for subgroup in subgroups:
        testsubset = testdata[maskforsubgroup(subgroup, testdata)]  # extract the correct subset of data
        trainsubset=traindata[maskforsubgroup(subgroup, traindata)]
        if len(trainsubset[target[-1]]) == 0:  # this should never happen
            # print(subgroup.data['Age in years'])
            print(subgroup.description.decryption, end='ERROR HERE\n')
            continue
        model = model_method(trainsubset[target[-1]], trainsubset[target[:-1]])
        model = model.fit()
        results = model.predict(testsubset[target[:-1]])
        indices = list(testsubset.index)
        for i, r in zip(indices, results):
            predictions[i].append(r)

    # we also do it for a model on all data
    model = model_method(traindata[target[-1]], traindata[target[:-1]])
    model = model.fit()
    results = model.predict(testdata[target[:-1]])
    indices = list(testdata.index)
    for i, r in zip(indices, results):
        predictions[i].append(r)

    average_predictions = [mean(l) for l in predictions]
    return average_predictions

def pattern_team_weighted_DTA(subgroups, testdata, traindata, target, model_method):

    if hasattr(model_method, '__call__'):
        pass
    else:
        raise ValueError(f"this is not a callable model: {model_method}")

    predictions = [[0, 0] for _ in testdata.index] # the first value sums all the predictions, the second the weights
    # make models and save the predictions for each model
    for subgroup in subgroups:
        testsubset = testdata[maskforsubgroup(subgroup, testdata)]  # extract the correct subset of data
        trainsubset=traindata[maskforsubgroup(subgroup, traindata)]
        model = model_method(trainsubset[target[-1]], trainsubset[target[:-1]])
        model = model.fit()
        results = model.predict(testsubset[target[:-1]])
        indices = list(testsubset.index)
        try:
            weight = 1/len(indices)
        except ZeroDivisionError:
            print(f'zerodivisionerror', end='')
            weight = 1  # this is just a filler value, as the model won't be used here anyways, as zero values are included.
        for i, r in zip(indices, results):
            predictions[i][0] += r
            predictions[i][1] += weight

    # we also do it for a model on all data
    model = model_method(traindata[target[-1]], traindata[target[:-1]])
    model = model.fit()
    results = model.predict(testdata[target[:-1]])
    indices = list(testdata.index)
    weight = 1/len(indices)
    for i, r in zip(indices, results):
        predictions[i][0] += r
        predictions[i][1] += weight

    average_predictions = [pred/W for pred, W in predictions]
    return average_predictions


def mean_error(True_values, predictions):
    l = [abs(t-p) for t, p in zip(True_values, predictions)]
    return mean(l)


def maskforsubgroup(subgroup, testset):
    description = subgroup.description.decryption
    masks = []
    try:
        for key in description:
            value = description[key]
            if type(value) == type([]):
                low = value[0]
                high = value[1]
                masks.append(high >= testset[key])
                masks.append(testset[key] >= low)
            else:
                masks.append(testset[key] == value)
    except ValueError:
        print(key)
        print(value)
        print(description)
        raise
    return [all(i) for i in zip(*masks)]
