"""Django command warpper for the overall process of training the ml model."""

from django.core.management.base import BaseCommand
import os
import tensorflow as tf
from datetime import datetime

from ml.training import (
    create_image_lists,
    create_module_graph,
    cache_bottlenecks,
    get_random_cached_bottlenecks,
    get_random_distorted_bottlenecks,
    should_distort_images,
    add_input_distortions,
    add_final_retrain_ops,
    add_evaluation_step,
    run_final_eval,
    save_graph_to_file,
    prepare_file_system,
    add_jpeg_decoding,
    logging_level_verbosity
)

import tensorflow_hub as hub


class Command(BaseCommand):
    """The ML model retrain command class."""

    help = "Retrain"

    MIN_NUM_IMAGES_REQUIRED_FOR_TRAINING = 10
    MIN_NUM_IMAGES_SUGGESTED_FOR_TRAINING = 100

    MIN_NUM_IMAGES_REQUIRED_FOR_TESTING = 3

    MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

    # path to folders of labeled images
    TRAINING_IMAGES_DIR = os.getcwd() + '/api/ml/training_images'

    # where to save the trained graph
    OUTPUT_GRAPH = os.getcwd() + '/api/ml/retrained_graph'

    # how many steps to store intermediate graph, if "0" then will not store
    INTERMEDIATE_STORE_FREQUENCY = 0

    # where to save the trained graph's labels
    OUTPUT_LABELS = os.getcwd() + '/api/ml/retrained_labels.txt'

    # where to save summary logs for TensorBoard
    TENSORBOARD_DIR = os.getcwd() + '/api/ml/tensorboard_logs'

    # how many training steps to run before ending
    # NOTE: original Google default is 4000, use 4000 (or possibly higher)
    # for production grade results
    HOW_MANY_TRAINING_STEPS = 4000

    # how large a learning rate to use when training
    LEARNING_RATE = 0.001

    # how often to evaluate the training results
    EVAL_STEP_INTERVAL = 5

    # how many images to train on at a time
    TRAIN_BATCH_SIZE = 100

    # How many images to test on. This test set is only used once, to evaluate
    # the final accuracy of the model after training completes.
    # A value of -1 causes the entire test set to be used,
    # which leads to more stable results across runs.
    TEST_BATCH_SIZE = -1

    # How many images to use in an evaluation batch. This validation set is used
    # much more often than the test set, and is an early indicator of how
    # accurate the model is during training. A value of -1 causes the entire
    # validation set to be used, which leads to
    # more stable results across training iterations,
    # but may be slower on large training sets.
    VALIDATION_BATCH_SIZE = -1

    # whether to print out a list of all misclassified test images
    PRINT_MISCLASSIFIED_TEST_IMAGES = False

    # Path to cache bottleneck layer values as files
    BOTTLENECK_DIR = os.getcwd() + '/api/ml/bottleneck_data'

    # whether to randomly flip half of the training images horizontally
    FLIP_LEFT_RIGHT = False

    # a percentage determining how much of a margin to randomly crop off the
    # training images
    RANDOM_CROP = 0

    # a percentage determining how much to randomly scale up the size of the
    # training images by
    RANDOM_SCALE = 0

    # a percentage determining how much to randomly multiply the training image
    # input pixels up or down by
    RANDOM_BRIGHTNESS = 0

    CHECKPOINT_NAME = '/tmp/_retrain_checkpoint'

    def add_arguments(self, parser):
        """Define the required arguments for the command."""
        parser.add_argument(
            '--size',
            type=str,
            help="slicing level (small or large), default is large"
        )

        parser.add_argument(
            '--intermediate_output_graphs_dir',
            type=str,
            default='/tmp/intermediate_graph/',
            help='Where to save the intermediate graphs.'
        )
        parser.add_argument(
            '--intermediate_store_frequency',
            type=int,
            default=0,
            help="""\
                How many steps to store intermediate graph. If "0" then will not
                store.\
            """
        )
        parser.add_argument(
            '--output_labels',
            type=str,
            default='/tmp/output_labels.txt',
            help='Where to save the trained graph\'s labels.'
        )
        parser.add_argument(
            '--summaries_dir',
            type=str,
            default='/tmp/retrain_logs',
            help='Where to save summary logs for TensorBoard.'
        )
        parser.add_argument(
            '--how_many_training_steps',
            type=int,
            default=4000,
            help='How many training steps to run before ending.'
        )
        parser.add_argument(
            '--learning_rate',
            type=float,
            default=0.001,
            help='How large a learning rate to use when training.'
        )
        parser.add_argument(
            '--testing_percentage',
            type=int,
            default=20,
            help='What percentage of images to use as a test set.'
        )
        parser.add_argument(
            '--validation_percentage',
            type=int,
            default=20,
            help='What percentage of images to use as a validation set.'
        )
        parser.add_argument(
            '--eval_step_interval',
            type=int,
            default=1,
            help='How often to evaluate the training results.'
        )
        parser.add_argument(
            '--train_batch_size',
            type=int,
            default=100,
            help='How many images to train on at a time.'
        )
        parser.add_argument(
            '--test_batch_size',
            type=int,
            default=-1,
            help="""\
            How many images to test on. This test set is only used once,
            to evaluate the final accuracy of the model after
            training completes.
            A value of -1 causes the entire test set to be used,
            which leads to more stable results across runs.\
            """
        )
        parser.add_argument(
            '--validation_batch_size',
            type=int,
            default=-1,
            help="""\
            How many images to use in an evaluation batch. This validation
            set is used much more often than the test set, and is an early
            indicator of how accurate the model is during training.
            A value of -1 causes the entire validation set to be used,
            which leads to more stable results across training iterations,
            but may be slower on large  training sets.\
            """
        )
        parser.add_argument(
            '--print_misclassified_test_images',
            default=False,
            help="""\
            Whether to print out a list of all misclassified test images.\
            """,
            action='store_true'
        )

        parser.add_argument(
            '--final_tensor_name',
            type=str,
            default='results',
            help="""\
            The name of the output classification layer in the retrained graph.\
            """
        )

        parser.add_argument(
            '--flip_left_right',
            default=False,
            help="""\
            Whether to randomly flip half of the training images horizontally.\
            """,
            action='store_true'
        )
        parser.add_argument(
            '--random_crop',
            type=int,
            default=0,
            help="""\
            A percentage determining how much
            of a margin to randomly crop off the training images.\
            """
        )
        parser.add_argument(
            '--random_scale',
            type=int,
            default=0,
            help="""\
            A percentage determining how much to randomly
            scale up the size of the training images by.\
            """
        )
        parser.add_argument(
            '--random_brightness',
            type=int,
            default=0,
            help="""\
            A percentage determining how much to randomly multiply
            the training image input pixels up or down by.\
            """
        )
        parser.add_argument(
            '--tfhub_module',
            type=str,
            default=('https://tfhub.dev/google/imagenet/%s' %
                     ('inception_v3/classification/3')),
            help="""\
            Which TensorFlow Hub module to use. For more options,
            search https://tfhub.dev for image feature vector modules.\
            """)
        parser.add_argument(
            '--saved_model_dir',
            type=str,
            default='./api/ml/trained_models/',
            help='Where to save the exported graph.')
        parser.add_argument(
            '--logging_verbosity',
            type=str,
            default='INFO',
            choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'],
            help='How much logging output should be produced.')

    def handle(self, *args, **options):
        """Do the retraining job."""
        print("starting program . . .")

        if options['size'] == 'small':
            options['image_dir'] = self.TRAINING_IMAGES_DIR + '/small'
            options['output_graph'] = self.OUTPUT_GRAPH + '_small.pb'
            self.FINAL_TENSOR_NAME = 'small'
            options['bottleneck_dir'] = self.BOTTLENECK_DIR + '/small'
        else:
            options['image_dir'] = self.TRAINING_IMAGES_DIR + '/large'
            options['output_graph'] = self.OUTPUT_GRAPH + '_large.pb'
            options['bottleneck_dir'] = self.BOTTLENECK_DIR + '/large'

        logging_verbosity = logging_level_verbosity(
            options['logging_verbosity']
        )
        tf.logging.set_verbosity(logging_verbosity)

        # Prepare necessary directories that can be used during training
        prepare_file_system(options['summaries_dir'],
                            options['intermediate_store_frequency'],
                            options['intermediate_output_graphs_dir'])

        # Look at the folder structure, and create lists of all the images.
        image_lists = create_image_lists(
            options['image_dir'],
            options['testing_percentage'],
            options['validation_percentage'])
        class_count = len(image_lists.keys())
        if class_count == 0:
            tf.logging.error(
                'No valid folders of images found at ' + options['image_dir'])
            return -1
        if class_count == 1:
            tf.logging.error(
                'Only one valid folder of images found at '
                + options['image_dir']
                + ' - multiple classes are needed for classification.'
            )
            return -1

        # See if the command-line flags mean we're applying any distortions.
        do_distort_images = should_distort_images(
            options['flip_left_right'],
            options['random_crop'],
            options['random_scale'],
            options['random_brightness'])

        # Set up the pre-trained graph.
        module_spec = hub.load_module_spec(options['tfhub_module'])
        graph, bottleneck_tensor, resized_image_tensor, wants_quantization = (
            create_module_graph(module_spec))

        # Add the new layer that we'll be training.
        with graph.as_default():
            (train_step, cross_entropy, bottleneck_input,
                ground_truth_input, final_tensor) = add_final_retrain_ops(
                    class_count, options['final_tensor_name'],
                    bottleneck_tensor, wants_quantization,
                    options['learning_rate'], is_training=True)

        with tf.Session(graph=graph) as sess:
            # Initialize all weights: for the module to their pretrained values,
            # and for the newly added retraining layer to random initial values.
            init = tf.global_variables_initializer()
            sess.run(init)

            # Set up the image decoding sub-graph.
            jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(
                module_spec
            )

            if do_distort_images:
                # We will be applying distortions, so set up the operations
                # we'll need.
                (distorted_jpeg_data_tensor,
                    distorted_image_tensor) = add_input_distortions(
                        options['flip_left_right'], options['random_crop'],
                        options['random_scale'], options['random_brightness'],
                        module_spec)
            else:
                # We'll make sure we've calculated the 'bottleneck'
                # image summaries and
                # cached them on disk.
                cache_bottlenecks(sess, image_lists, options['image_dir'],
                                  options['bottleneck_dir'], jpeg_data_tensor,
                                  decoded_image_tensor, resized_image_tensor,
                                  bottleneck_tensor, options['tfhub_module'])

            # Create the operations we need to evaluate the
            # accuracy of our new layer.
            evaluation_step, _ = add_evaluation_step(
                final_tensor,
                ground_truth_input
            )

            # Merge all the summaries and write them out to the summaries_dir
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(
                options['summaries_dir'] + '/train', sess.graph)

            validation_writer = tf.summary.FileWriter(
                options['summaries_dir'] + '/validation')

            # Create a train saver that is used to restore values
            # into an eval graph when exporting models.
            train_saver = tf.train.Saver()

            # Run the training for as many cycles
            # as requested on the command line.
            for i in range(options['how_many_training_steps']):
                # Get a batch of input bottleneck values, either calculated
                # fresh every time with distortions applied,
                # or from the cache stored on disk.
                if do_distort_images:
                    (train_bottlenecks,
                        train_ground_truth) = get_random_distorted_bottlenecks(
                        sess, image_lists, options['train_batch_size'],
                        'training', options['image_dir'],
                        distorted_jpeg_data_tensor,
                        distorted_image_tensor, resized_image_tensor,
                        bottleneck_tensor)
                else:
                    (train_bottlenecks,
                        train_ground_truth, _) = get_random_cached_bottlenecks(
                        sess, image_lists, options['train_batch_size'],
                        'training', options['bottleneck_dir'],
                        options['image_dir'], jpeg_data_tensor,
                        decoded_image_tensor, resized_image_tensor,
                        bottleneck_tensor, options['tfhub_module'])

                # Feed the bottlenecks and ground truth into the graph,
                # and run a training step. Capture training summaries for
                # TensorBoard with the `merged` op.
                train_summary, _ = sess.run(
                    [merged, train_step],
                    feed_dict={
                        bottleneck_input: train_bottlenecks,
                        ground_truth_input: train_ground_truth
                    })
                train_writer.add_summary(train_summary, i)

                # Every so often, print out how well the graph is training.
                is_last_step = (i + 1 == options['how_many_training_steps'])
                if (i % options['eval_step_interval']) == 0 or is_last_step:
                    train_accuracy, cross_entropy_value = sess.run(
                        [evaluation_step, cross_entropy],
                        feed_dict={
                            bottleneck_input: train_bottlenecks,
                            ground_truth_input: train_ground_truth})
                    tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                                    (datetime.now(), i, train_accuracy * 100))
                    tf.logging.info('%s: Step %d: Cross entropy = %f' %
                                    (datetime.now(), i, cross_entropy_value))
                    # TODO: Make this use an eval graph, to avoid quantization
                    # moving averages being updated by the validation set,
                    # though in practice this makes a negligable difference.
                    validation_bottlenecks, validation_ground_truth, _ = (
                        get_random_cached_bottlenecks(
                            sess, image_lists, options['validation_batch_size'],
                            'validation', options['bottleneck_dir'],
                            options['image_dir'], jpeg_data_tensor,
                            decoded_image_tensor, resized_image_tensor,
                            bottleneck_tensor, options['tfhub_module']))
                    # Run a validation step and capture training
                    # summaries for TensorBoard
                    # with the `merged` op.
                    validation_summary, validation_accuracy = sess.run(
                        [merged, evaluation_step],
                        feed_dict={
                            bottleneck_input: validation_bottlenecks,
                            ground_truth_input: validation_ground_truth})
                    validation_writer.add_summary(validation_summary, i)
                    tf.logging.info(
                        '%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                        (
                            datetime.now(),
                            i, validation_accuracy * 100,
                            len(validation_bottlenecks)
                        )
                    )

                # Store intermediate results
                intermediate_frequency = options['intermediate_store_frequency']

                if (intermediate_frequency > 0
                    and (i % intermediate_frequency == 0)
                    and i > 0):
                    # If we want to do an intermediate save,
                    # save a checkpoint of the train
                    # graph, to restore into the eval graph.
                    train_saver.save(sess, self.CHECKPOINT_NAME)
                    intermediate_file_name = (
                        options['intermediate_output_graphs_dir']
                        + 'intermediate_' + str(i) + '.pb')
                    tf.logging.info('Save intermediate result to : '
                                    + intermediate_file_name)
                    save_graph_to_file(
                        intermediate_file_name,
                        module_spec,
                        class_count,
                        options['final_tensor_name'],
                        options['learning_rate']
                    )

            # After training is complete, force one last
            # save of the train checkpoint.
            train_saver.save(sess, self.CHECKPOINT_NAME)

            # We've completed all our training,
            # so run afinal test evaluation on
            # some new images we haven't used before.
            run_final_eval(
                sess, module_spec, class_count, image_lists,
                jpeg_data_tensor, decoded_image_tensor, resized_image_tensor,
                bottleneck_tensor, options['test_batch_size'],
                options['bottleneck_dir'], options['image_dir'],
                options['tfhub_module'],
                options['print_misclassified_test_images'],
                options['final_tensor_name'],
                options['learning_rate']
            )

            # Write out the trained graph and labels with the weights stored as
            # constants.
            tf.logging.info('Save final result to : ' + options['output_graph'])
            if wants_quantization:
                tf.logging.info(
                    'The model is instrumented for quantization with TF-Lite'
                )
            save_graph_to_file(
                options['output_graph'],
                module_spec,
                class_count,
                options['final_tensor_name'],
                options['learning_rate']
            )
            with tf.gfile.GFile(options['output_labels'], 'w') as f:
                f.write('\n'.join(image_lists.keys()) + '\n')
