for i in 95
# 96 97 98
do
    # train
    python3 main.py -m train -r ./record/"$i" -c ./record/"$i"/config.ini -k ./record/"$i"/best_model.pth -o ./record/"$i"/optimizer.pth
    # test
    python3 main.py -m test -r ./record/"$i" -c ./record/"$i"/config.ini
    # count
    # python3 ./prediction-result-presentation/analyze.py -tf ./data/raw_data/SWGIM_year -f ./record/"$i"/prediction_frame1.csv \
    #         -r ./record/"$i"
    # multiple frame output
    for f in 6
    do
        python3 ./prediction-result-presentation/analyze.py -tf ./data/raw_data/SWGIM_year -f ./record/"$i"/prediction_frame"$f".csv \
                -r ./record/"$i" -o record_frame"$f".json
    done
done
