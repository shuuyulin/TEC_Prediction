for i in 151
do
    # tests
    python3 main.py -m test -r ./record/"$i" -c ./record/"$i"/config.ini -t
    # count
    for f in 1
    do
        python3 ./prediction-result-presentation/analyze.py -tf ./data/raw_data/SWGIM_year -f ./record/"$i"/train_prediction_frame"$f".csv \
                -r ./record/"$i" -o train_record_frame"$f".json
    done
    rm ./record/"$i"/train_prediction_frame"$f".csv 
done
