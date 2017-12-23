import numpy as np
import theano
import theano.tensor as T

def load_testing(size=5, length=10000, classes=3):
    # Super-duper important: set a seed so you always have the same data over multiple runs.
    np.random.seed(123)

    # Generate random data between 0 and 1 as a numpy array.
    # The size determines the amount of input values.
    x=[]
    for i in range (0, length):
        x.append(np.asarray(np.random.uniform(low=0, high=1, size=size), dtype='float64'))

    # Split up the input array into training/test/validation sets.
    train_set_x = x[:int(len(x)*0.6)]
    test_set_x = x[int(len(x)*0.6):int(len(x)*0.8)]
    valid_set_x = x[int(len(x)*0.8):]

    # For each input in x, we will generate a class for y.
    # If classes is set to less than or equal to 1, it will output real numbers
    # for regression.
    y=[]
    for row in x:
        row_tmp = list()
        for i in range(0, len(row)):
            # Normalize from -1 to 1 and take the hyperbolic tangent of the separate values.
            # Note: we do "* (i+1)" on each value to give a different importance to the input variables.
            row_tmp.append(row[i] * 2. - 1)
            row_tmp[i] = (np.tanh(row_tmp[i])) * (i+1)

        output = np.sin(np.mean(row_tmp))

        # If classes > 1 we will output classes, else we output real numbers.
        if (classes > 1):
            classnum = 0

            # Generate class limits depending on the amount of classes you want.
            classranges = np.arange(-1, 1, (2./float(classes)))[::-1]
            for i in classranges:
                if (output >= i):
                    y.append(classnum)
                    break
                classnum = classnum + 1
        else:
            y.append(output)

    # Convert Y to a numpy array and split it up into train/test/validation sets.
    y = np.asarray(y)
    train_set_y = y[:int(len(x)*0.6)]
    test_set_y = y[int(len(x)*0.6):int(len(x)*0.8)]
    valid_set_y = y[int(len(x)*0.8):]

    def shared_dataset(data_xy, borrow=True):
        #Create a shared dataset. This is only if you are using Theano
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')

    if (use_theano):
        # Convert the numpy arrays into theano datasets, IF using theano.
        train_set_x, train_set_y = shared_dataset((train_set_x, train_set_y))
        test_set_x, test_set_y = shared_dataset((test_set_x, test_set_y))
        valid_set_x, valid_set_y = shared_dataset((valid_set_x, valid_set_y))

    # Return the dataset in pairs of x,y for train/test/validation sets.
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
