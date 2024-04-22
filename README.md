# csc413-football-prediction

- Run `python join_train_val.py` to create the data_recent_and_val.csv
- Run `python compute_berrar_ratings.py` to get the berrar ratings
- To train model with command line:
```
python3 main.py train --model-type lstm \
                      --device gpu \
                      --save-model True \
                      --inception-out-dim 24 \
                      --inception-bottleneck-dim None \
                      --epochs 6 \
                      --lr 0.0005 \
                      --gradient-accumulation 10
```
- To test the trained model:
```
python3 main.py test --device gpu --model-path trained_models/lstm_rps_0.0005.pth
```