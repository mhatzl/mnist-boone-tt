import boone.NetFactory;
import boone.NeuralNet;
import boone.PatternSet;
import boone.Trainer;
import boone.io.BooneFilter;
import boone.training.RpropTrainer;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class Main {

    final static double TARGET_CLASS_DELTA = 0.0000000000001D;

    /**
     * Runs over all training and test patterns of the mnist dataset by training a ANN annSteps*annEpochs times before testing
     * @param mnistANN file to a saved Boone-ANN in xnet-format, if the file doesn't exist, a new ANN is created
     * @param classTargetValues array of values a output neuron is trained to for classification
     * @param nonClassTargetValues array of values a output neuron is trained to for non-classification
     * @param annSteps number of steps a ANN gets trained
     * @param annEpochs number of epochs a ANN gets trained
     * @param nrHiddenNeurons number of hidden neurons used when first creating a ANN
     * @return TT_TestResults containing information about test results of this neuralNetRun
     */
    public static TT_TestResults neuralNetRun(File mnistANN, double[] classTargetValues, double[] nonClassTargetValues, int annSteps, int annEpochs, int nrHiddenNeurons) {
        return neuralNetRun(mnistANN, classTargetValues, nonClassTargetValues, annSteps, annEpochs, nrHiddenNeurons, false);
    }

    /**
     * Runs over all training and test patterns of the mnist dataset by training a ANN annSteps*annEpochs times before testing
     * @param mnistANN file to a saved Boone-ANN in xnet-format, if the file doesn't exist, a new ANN is created
     * @param classTargetValues array of values a output neuron is trained to for classification
     * @param nonClassTargetValues array of values a output neuron is trained to for non-classification
     * @param annSteps number of steps a ANN gets trained
     * @param annEpochs number of epochs a ANN gets trained
     * @param nrHiddenNeurons number of hidden neurons used when first creating a ANN
     * @param debug true: additional debug information is printed
     * @return TT_TestResults containing information about test results of this neuralNetRun
     */
    public static TT_TestResults neuralNetRun(File mnistANN, double[] classTargetValues, double[] nonClassTargetValues, int annSteps, int annEpochs, int nrHiddenNeurons, boolean debug) {
        NeuralNet net;
        MnistMatrix[] mnistMatricesTrain;
        MnistMatrix[] mnistMatricesTest;

        if ((classTargetValues.length != 10) || (nonClassTargetValues.length != 10)) {
            System.out.println("Please provide 10 target values for class and non-class");
            return null;
        }

        try {
            // get training data
            mnistMatricesTrain = new MnistDataReader().readData("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
            // get test data
            mnistMatricesTest = new MnistDataReader().readData("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");
        } catch (IOException exception) {
            System.out.println("Couldn't load mnist datasets");
            return null;
        }

        if (mnistANN.exists()) {
            var filter = new BooneFilter();
            try {
                filter.setCompressed(true);
                net = NeuralNet.load(mnistANN, filter);
            } catch (Exception exception) {
                try {
                    filter.setCompressed(false);
                    net = NeuralNet.load(mnistANN, filter);
                } catch (Exception ex) {
                    System.out.println("Couldn't load ANN " + mnistANN);
                    return null;
                }
            }
        } else {
            net = NetFactory.createFeedForward(new int[]{784, nrHiddenNeurons, 10}, false, new boone.map.Function.Sigmoid(), new RpropTrainer(), null, null);
        }

        PatternSet trainPatterns = getMnistPatternSet(classTargetValues, nonClassTargetValues, mnistMatricesTrain);
        PatternSet testPatterns = getMnistPatternSet(classTargetValues, nonClassTargetValues, mnistMatricesTest);

        Trainer trainer = net.getTrainer();
        trainer.setTrainingData(trainPatterns);
        trainer.setTestData(testPatterns);
        trainer.setEpochs(annEpochs);
        trainer.setStepMode(true);

        System.out.println("*** Training " + (annSteps * annEpochs) + " steps * epochs ...");
        for (int i = 0; i < annSteps; i++) {
            trainer.train();
            System.out.println((i * annEpochs));
        }

        System.out.println("\n*** Testing the network...");
        double[] testsError = new double[testPatterns.size()];
        boolean[] testsSuccessful = new boolean[testPatterns.size()];

        for (int i = 0; i < testPatterns.size(); i++) {
            testsError[i] = net.getTrainer().test(testPatterns.getInputs().get(i), testPatterns.getTargets().get(i));

            if (debug) {
                System.out.println("** Testing pattern " + i);
                System.out.println("Inputsize = " + testPatterns.getInputs().get(i).size());
                System.out.println("For input \n" + formattedInputToString(testPatterns.getInputs().get(i)));
                System.out.println("Target " + " = " + testPatterns.getTargets().get(i));
                for (int j = 0; j < net.getOutputNeuronCount(); j++) {
                    System.out.println("Output " + j + " = " + net.getOutputNeuron(j).getOutput());
                }
                System.out.println("Error " + i + " = " + testsError[i]);
            }

            testsSuccessful[i] = wasTestSuccessful(testPatterns.getTargets().get(i), net.getOutput(null), classTargetValues);
        }

        int successCnt = 0;
        for (var testSuccessful : testsSuccessful) {
            if (testSuccessful) {
                successCnt++;
            }
        }

        try {
            net.save(mnistANN);
        } catch (IOException exception) {
            System.out.println("Couldn't save ANN " + mnistANN);
        }

        System.out.println("NeuralNetRun Done.");

        TT_TestResults testResults = new TT_TestResults();
        testResults.testsError = testsError;
        testResults.successRate = (double)successCnt / testPatterns.size();
        testResults.testsSuccessful = testsSuccessful;

        return testResults;
    }

    /**
     * Get mnist patternSet to be used by boone
     * @param classTargetValues array of values a output neuron is trained to for classification
     * @param nonClassTargetValues array of values a output neuron is trained to for non-classification
     * @param mnistMatrices matrices with mnist data
     * @return patternSet containing the mnist data in a form boone can use
     */
    private static PatternSet getMnistPatternSet(double[] classTargetValues, double[] nonClassTargetValues, MnistMatrix[] mnistMatrices) {
        PatternSet patternSet = new PatternSet();
        for (var mnistMatrix : mnistMatrices) {
            MnistPatternSet mnistSet = new MnistPatternSet(mnistMatrix, classTargetValues, nonClassTargetValues);

            patternSet.getInputs().add(mnistSet.getInputs());
            patternSet.getTargets().add(mnistSet.getTargets());
        }
        return patternSet;
    }

    /**
     * Method to check if a test was successful
     * @param targets targets for this test
     * @param outputs outputs of the ANN for this test
     * @param classTargetValues used classTargetValues of the ANN
     * @return true: the test was successful
     */
    public static boolean wasTestSuccessful(List<Double> targets, double[] outputs, double[] classTargetValues) {
        for (int j = 0; j < targets.size(); j++) {
            if (Math.abs(targets.get(j) - classTargetValues[j]) < TARGET_CLASS_DELTA) {
                double actOutputClassDistance = Math.abs(outputs[j] - classTargetValues[j]);
                for (int x = 0; x < outputs.length; x++) {
                    double otherClassDistance = Math.abs(outputs[x] - classTargetValues[x]);
                    if (x != j && (actOutputClassDistance > otherClassDistance)) {
                        return false;
                    }
                }
            }
        }
        return true;
    }


    public static void main(String[] args) {
        double[] classTargetValues = new double[]{1.0D, 0.9D, 1.0D, 0.9D, 1.0D, 0.9D, 1.0D, 0.9D, 1.0D, 0.9D}; // index 0 = digit 0, index 1 = digit 1, ...
        double[] nonClassTargetValues = new double[]{0.1D, 0.0D, 0.1D, 0.0D, 0.1D, 0.0D, 0.1D, 0.0D, 0.1D, 0.0D};
        File mnistANN = new File("mnistANN_784i_100h_10o_classical.xnet");
        int steps = 1;
        int epochs = 2;
        int nrHiddenNeurons = 100;

        var startTime = System.nanoTime();

        TT_TestResults testResults = neuralNetRun(mnistANN, classTargetValues, nonClassTargetValues, steps, epochs, nrHiddenNeurons);
        System.out.println(testResults);

        System.out.println("Run took: " + ((System.nanoTime() - startTime) / 1000000000.0D) + "s");
    }

    /**
     * Class containing useful test results
     */
    private static class TT_TestResults {
        public double [] testsError;
        public boolean[] testsSuccessful;
        public double successRate;

        public String toString() {
            StringBuilder s = new StringBuilder("Tests successful:\n");
            for (int i = 0; i < 4; i++) {
                s.append("pattern ").append(i).append(": ").append(testsSuccessful[i]).append("\n");
            }

            return "SuccessRate = " + successRate + "\n" + s;
        }
    }

    private static String formattedInputToString(List<Double> inputs) {
        StringBuilder s = new StringBuilder();

        for (int i = 1; i <= inputs.size(); i++) {
            s.append(inputs.get(i - 1)).append(" ");
            if (i % 28 == 0)
                s.append("\n");
        }
        return s.toString();
    }

    private static void printMnistMatrix(final MnistMatrix matrix) {
        System.out.println("label: " + matrix.getLabel());
        for (int r = 0; r < matrix.getNumberOfRows(); r++ ) {
            for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
                System.out.print(matrix.getValue(r, c) + " ");
            }
            System.out.println();
        }
    }
}
