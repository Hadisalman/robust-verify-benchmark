python main.py --model mnist_cnn_small --epsilon 0.1 --training-mode ADV &&
python main.py --model mnist_cnn_small --epsilon 0.1 --training-mode NOR &&
python main.py --model mnist_cnn_small --epsilon 0.1 --training-mode LPD &&

python main.py --model mnist_cnn_wide_1 --epsilon 0.1 --training-mode ADV &&
python main.py --model mnist_cnn_wide_1 --epsilon 0.1 --training-mode NOR &&
python main.py --model mnist_cnn_wide_1 --epsilon 0.1 --training-mode LPD &&

python main.py --model mnist_cnn_wide_2 --epsilon 0.1 --training-mode ADV &&
python main.py --model mnist_cnn_wide_2 --epsilon 0.1 --training-mode NOR &&
python main.py --model mnist_cnn_wide_2 --epsilon 0.1 --training-mode LPD &&

python main.py --model mnist_cnn_wide_4 --epsilon 0.1 --training-mode ADV &&
python main.py --model mnist_cnn_wide_4 --epsilon 0.1 --training-mode NOR &&
python main.py --model mnist_cnn_wide_4 --epsilon 0.1 --training-mode LPD &&

python main.py --model mnist_cnn_wide_8 --epsilon 0.1 --training-mode ADV &&
python main.py --model mnist_cnn_wide_8 --epsilon 0.1 --training-mode NOR &&

python main.py --model mnist_cnn_deep_1 --epsilon 0.1 --training-mode ADV &&
python main.py --model mnist_cnn_deep_1 --epsilon 0.1 --training-mode NOR &&
python main.py --model mnist_cnn_deep_1 --epsilon 0.1 --training-mode LPD &&

python main.py --model mnist_cnn_deep_2 --epsilon 0.1 --training-mode ADV &&
python main.py --model mnist_cnn_deep_2 --epsilon 0.1 --training-mode NOR &&

python main.py --model MLP_9_500 --epsilon 0.1 --training-mode ADV &&
python main.py --model MLP_9_500 --epsilon 0.1 --training-mode NOR &&

python main.py --model MLP_9_100 --epsilon 0.1 --training-mode ADV &&
python main.py --model MLP_9_100 --epsilon 0.1 --training-mode NOR &&
python main.py --model MLP_9_100 --epsilon 0.1 --training-mode LPD &&

python main.py --model MLP_2_100 --epsilon 0.1 --training-mode ADV &&
python main.py --model MLP_2_100 --epsilon 0.1 --training-mode NOR &&
python main.py --model MLP_2_100 --epsilon 0.1 --training-mode LPD