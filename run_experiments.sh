for i in 84 85 86
do
    # train
    python3 main.py -m train -r ./record/"$i" -c ./record/"$i"/config.ini
    # test
    python3 main.py -m test -r ./record/"$i" -c ./record/"$i"/config.ini
    # count
    python3 ./prediction-result-presentation/analyze.py -m global -tf ./data/raw_data/SWGIM_year -f ./record/"$i"/prediction.csv \
            -r ./record/"$i"
done
