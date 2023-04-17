for i in 151
do
    # train
    python3 main.py -m train -r ./record/"$i" -c ./record/"$i"/config.ini
    #  -k ./record/"$i"/best_model.pth -o ./record/"$i"/optimizer.pth
    # test
    python3 main.py -m test -r ./record/"$i" -c ./record/"$i"/config.ini
    #  -s 24
    # multiple frames output
    for f in 1
    do
        python3 ./prediction-result-presentation/analyze.py -tf ./data/raw_data/SWGIM_year -f ./record/"$i"/prediction_frame"$f".csv \
                -r ./record/"$i" -o record_frame"$f".json
    done
done
