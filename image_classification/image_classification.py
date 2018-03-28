#                                                            _
# image_classification ds app
#
# (c) 2016 Fetal-Neonatal Neuroimaging & Developmental Science Center
#                   Boston Children's Hospital
#
#              http://childrenshospital.org/FNNDSC/
#                        dev@babyMRI.org
#

import os
import tensorflow as tf
import mnist_reader
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# import the Chris app superclass
from chrisapp.base import ChrisApp


class Image_classification(ChrisApp):
    """
    Classify medical mage data.
    """
    AUTHORS         = 'Anik (bhattacharjee.an@husky.neu.edu)'
    SELFPATH        = os.path.dirname(os.path.abspath(__file__))
    SELFEXEC        = os.path.basename(__file__)
    EXECSHELL       = 'python3'
    TITLE           = 'My app'
    CATEGORY        = 'classification'
    TYPE            = 'ds'
    DESCRIPTION     = 'Classify medical mage data'
    DOCUMENTATION   = 'http://wiki'
    VERSION         = '0.1'
    LICENSE         = 'Opensource (MIT)'
    MAX_NUMBER_OF_WORKERS = 1  # Override with integer value
    MIN_NUMBER_OF_WORKERS = 1  # Override with integer value
    MAX_CPU_LIMIT         = '' # Override with millicore value as string, e.g. '2000m'
    MIN_CPU_LIMIT         = '' # Override with millicore value as string, e.g. '2000m'
    MAX_MEMORY_LIMIT      = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_MEMORY_LIMIT      = '' # Override with string, e.g. '1Gi', '2000Mi'
    MIN_GPU_LIMIT         = 0  # Override with the minimum number of GPUs, as an integer, for your plugin
    MAX_GPU_LIMIT         = 0  # Override with the maximum number of GPUs, as an integer, for your plugin

    # Fill out this with key-value output descriptive info (such as an output file path
    # relative to the output dir) that you want to save to the output meta file when
    # called with the --saveoutputmeta flag
    OUTPUT_META_DICT = {}
 
    def define_parameters(self):
        """
        Define the CLI arguments accepted by this plugin app.
        """

    def run(self, options):
        """
        Define the code to be run by this plugin app.
        """

        # Disabling architechture related warnings from Tensorflow
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        batch_size = 100
        number_of_dimensions = 784
        number_of_classes = 10
        training_set_size = 60000
        display_step = 100
        output_file = options.outputdir + "/results.txt"
        file = open(output_file, 'w')

        training_images, training_labels = mnist_reader.load_mnist(options.inputdir, 
                                                                    kind='train')
        test_images, test_labels = mnist_reader.load_mnist(options.inputdir, 
                                                                    kind='t10k')

        encoder = OneHotEncoder()
        labels = np.array(list(training_labels) + list(test_labels)).reshape(-1, 1)
        labels = encoder.fit_transform(labels).todense()
        training_labels = labels[:training_set_size]
        test_labels = labels[training_set_size:]

        session = tf.InteractiveSession()
        x = tf.placeholder(tf.float32, shape=[None, number_of_dimensions])
        y = tf.placeholder(tf.float32, shape=[None, number_of_classes])

        w = tf.Variable(tf.zeros(shape=[number_of_dimensions ,number_of_classes]))
        b = tf.Variable(tf.zeros(shape=[number_of_classes]))

        session.run(tf.global_variables_initializer())  

        y_ = tf.matmul(x,w) + b

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, 
                                                                            logits=y_))

        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        file.write("Training model....\n")
        for index in range(1000):
            start_index = (index * batch_size) % len(training_images)
            end_index = ((index + 1) * batch_size) % len(training_images)
            batch_x = training_images[start_index: end_index]
            _, l = session.run([train_step, cross_entropy], 
                    feed_dict={x:batch_x, y:training_labels[start_index:end_index]})
            
            if index % display_step == 0:
                line = "Step: " + str(index) + " Minibatch cross entropy: " + str(l) +"\n"
                file.write(line)

        file.write("\nDone training model\n")
        file.write("Testing model...\n")
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        line = "Accuracy of prediction by trained model: " +  str(accuracy.eval(feed_dict={x:test_images, y:test_labels}) )
        file.write(line)
        file.close()


# ENTRYPOINT
if __name__ == "__main__":
    app = Image_classification()
    app.launch()
