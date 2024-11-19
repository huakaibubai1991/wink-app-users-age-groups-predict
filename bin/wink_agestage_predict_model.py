from load_util import get_qid_data
import lightgbm as lgb
import numpy
import sys

def save_file(save_path, gids, y_prob):
    y_pred = numpy.argmax(y_prob, axis=1)
    with open(save_path, 'w') as output_file:
        for i,g in enumerate(gids):
            predict = y_pred[i]
            y_prob_round = [round(x, 5) for x in y_prob[i]]
            line = str(g) + "\t" + str(predict) + "\t" + str(y_prob_round)+ "\n"
            output_file.write(line)
    print(len(y_pred))
    print(len(gids))
    print(len(y_prob))
    return

def main():
    if (len(sys.argv) > 1):
        test_data_path = sys.argv[1]
        model_path = sys.argv[2]
        result_path = sys.argv[3]

    # read data
    X, gid = get_qid_data(test_data_path)
    print(gid)
    # get model and predict data
    model = lgb.Booster(model_file=model_path)
    y_prob = model.predict(X)
    #print(gid, y_prob)
    print(len(y_prob))
    print(X.shape)
    print(len(gid))
   
    # save result data
    save_file(result_path, gid, y_prob)


if __name__ == '__main__':
    main()


