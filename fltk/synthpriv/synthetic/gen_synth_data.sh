/usr/bin/time -v python -u evaluate.py --target-variable='income' --train-data-path=./data/adult_processed_train.csv --test-data-path=./data/adult_processed_test.csv --normalize-data ron-gauss --enable-privacy --target-epsilon=2 --save-synthetic --output-data-path ~/code/fltk-testbed-gr-1/data/
/usr/bin/time -v python -u evaluate.py --target-variable='income' --train-data-path=./data/adult_processed_train.csv --test-data-path=./data/adult_processed_test.csv --normalize-data ron-gauss --enable-privacy --target-epsilon=5 --save-synthetic --output-data-path ~/code/fltk-testbed-gr-1/data/
/usr/bin/time -v python -u evaluate.py --target-variable='income' --train-data-path=./data/adult_processed_train.csv --test-data-path=./data/adult_processed_test.csv --normalize-data ron-gauss --enable-privacy --target-epsilon=8 --save-synthetic --output-data-path ~/code/fltk-testbed-gr-1/data/
/usr/bin/time -v python -u evaluate.py --target-variable='income' --train-data-path=./data/adult_processed_train.csv --test-data-path=./data/adult_processed_test.csv --normalize-data imle --sigma=0.8 --enable-privacy --target-epsilon=2 --save-synthetic --output-data-path ~/code/fltk-testbed-gr-1/data/
/usr/bin/time -v python -u evaluate.py --target-variable='income' --train-data-path=./data/adult_processed_train.csv --test-data-path=./data/adult_processed_test.csv --normalize-data imle --sigma=0.7 --enable-privacy --target-epsilon=5 --save-synthetic --output-data-path ~/code/fltk-testbed-gr-1/data/
/usr/bin/time -v python -u evaluate.py --target-variable='income' --train-data-path=./data/adult_processed_train.csv --test-data-path=./data/adult_processed_test.csv --normalize-data imle --sigma=0.6 --enable-privacy --target-epsilon=8 --save-synthetic --output-data-path ~/code/fltk-testbed-gr-1/data/
/usr/bin/time -v python -u evaluate.py --target-variable='income' --train-data-path=./data/adult_processed_train.csv --test-data-path=./data/adult_processed_test.csv --normalize-data pate-gan --lap-scale=1e-4 --enable-privacy --target-epsilon=2 --save-synthetic --output-data-path ~/code/fltk-testbed-gr-1/data/
/usr/bin/time -v python -u evaluate.py --target-variable='income' --train-data-path=./data/adult_processed_train.csv --test-data-path=./data/adult_processed_test.csv --normalize-data pate-gan --lap-scale=3e-4 --enable-privacy --target-epsilon=5 --save-synthetic --output-data-path ~/code/fltk-testbed-gr-1/data/
/usr/bin/time -v python -u evaluate.py --target-variable='income' --train-data-path=./data/adult_processed_train.csv --test-data-path=./data/adult_processed_test.csv --normalize-data pate-gan --lap-scale=3e-4 --enable-privacy --target-epsilon=8 --save-synthetic --output-data-path ~/code/fltk-testbed-gr-1/data/
/usr/bin/time -v python -u evaluate.py --target-variable='income' --train-data-path=./data/adult_processed_train.csv --test-data-path=./data/adult_processed_test.csv --normalize-data dp-wgan --sigma=1.0 --enable-privacy --target-epsilon=2 --save-synthetic --output-data-path ~/code/fltk-testbed-gr-1/data/
/usr/bin/time -v python -u evaluate.py --target-variable='income' --train-data-path=./data/adult_processed_train.csv --test-data-path=./data/adult_processed_test.csv --normalize-data dp-wgan --sigma=0.9 --enable-privacy --target-epsilon=5 --save-synthetic --output-data-path ~/code/fltk-testbed-gr-1/data/
/usr/bin/time -v python -u evaluate.py --target-variable='income' --train-data-path=./data/adult_processed_train.csv --test-data-path=./data/adult_processed_test.csv --normalize-data dp-wgan --sigma=0.8 --enable-privacy --target-epsilon=8 --save-synthetic --output-data-path ~/code/fltk-testbed-gr-1/data/