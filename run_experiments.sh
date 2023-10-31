for i in 157 158 159 160 161 162 163 164 165
do
    python3 -m src.main -r ./record/"$i" -c ./record/"$i"/config.ini
    # test
    python3 -m src.main -m test -r ./record/"$i" -c ./record/"$i"/config.ini
    
    python3 -m src.analyze_tools.analyze -tf ./data/SWGIM3.0_year\
            -f ./record/"$i"/prediction_frame1.csv \
            -r ./record/"$i" -o record.json
done
