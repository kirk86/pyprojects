jobdispatch --env=THEANO_FLAGS=device=gpu,floatX=float32,force_device=True --duree=3h --mem=6G --gpu $SCRATCH/results/forgetting/random_search_dropout_maxout_mnist_amazon/worker.sh $SCRATCH/results/forgetting/random_search_dropout_maxout_mnist_amazon/exp/"{{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24}}"
