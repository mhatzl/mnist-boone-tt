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

    public static TT_TestResults neuralNetRun(File mnistANN, double classTargetValue, double nonClassTargetValue, int annSteps, int annEpochs, int nrHiddenLayers) {
        return neuralNetRun(mnistANN, classTargetValue, nonClassTargetValue, annSteps, annEpochs, nrHiddenLayers, false);
    }

    public static TT_TestResults neuralNetRun(File mnistANN, double classTargetValue, double nonClassTargetValue, int annSteps, int annEpochs, int nrHiddenLayers, boolean debug) {
        final double TARGET_CLASS_DELTA = 0.0000000000001D;
        NeuralNet net;
        MnistMatrix[] mnistMatricesTrain;
        MnistMatrix[] mnistMatricesTest;

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
            try {
                net = NeuralNet.load(mnistANN, new BooneFilter());
            } catch (IOException exception) {
                System.out.println("Couldn't load ANN " + mnistANN);
                return null;
            }
        } else {
            net = NetFactory.createFeedForward(new int[]{784, nrHiddenLayers, 10}, false, new boone.map.Function.Sigmoid(), new RpropTrainer(), null, null);
        }

        PatternSet trainPatterns = new PatternSet();
        for (var mnistMatrix : mnistMatricesTrain) {
            MnistPatternSet mnistSet = new MnistPatternSet(mnistMatrix, classTargetValue, nonClassTargetValue);

            trainPatterns.getInputs().add(mnistSet.getInputs());
            trainPatterns.getTargets().add(mnistSet.getTargets());
        }
        PatternSet testPatterns = new PatternSet();
        for (var mnistMatrix : mnistMatricesTest) {
            MnistPatternSet mnistSet = new MnistPatternSet(mnistMatrix, classTargetValue, nonClassTargetValue);

            testPatterns.getInputs().add(mnistSet.getInputs());
            testPatterns.getTargets().add(mnistSet.getTargets());
        }

        Trainer trainer = net.getTrainer();
        trainer.setTrainingData(trainPatterns);
        trainer.setTestData(testPatterns);
        trainer.setEpochs(annEpochs);
        trainer.setStepMode(true);

        System.out.println("*** Training " + (annSteps * annEpochs) + " epochs...");
        for (int i = 0; i < annSteps; i++) {
            trainer.train();
            System.out.println((i * annEpochs));
        }


        System.out.println("\n*** Testing the network...");
        double[] testsError = new double[testPatterns.size()];
        boolean[] testsSuccessful = new boolean[testPatterns.size()];

        for (int i = 0; i < testPatterns.size(); i++) {
            System.out.println("** Testing pattern " + i);
            testsError[i] = net.getTrainer().test(testPatterns.getInputs().get(i), testPatterns.getTargets().get(i));

            if (debug) {
                System.out.println("Inputsize = " + testPatterns.getInputs().get(i).size());
                System.out.println("For input \n" + formattedInputToString(testPatterns.getInputs().get(i)));
                System.out.println("Target " + " = " + testPatterns.getTargets().get(i));
                for (int j = 0; j < net.getOutputNeuronCount(); j++) {
                    System.out.println("Output " + j + " = " + net.getOutputNeuron(j).getOutput());
                }
                System.out.println("Error " + i + " = " + testsError[i]);
            }

            // detect if test was successful
            testsSuccessful[i] = true;
            List<Double> targets = testPatterns.getTargets().get(i);
            for (int j = 0; j < targets.size(); j++) {
                if (Math.abs(targets.get(j) - classTargetValue) < TARGET_CLASS_DELTA) {
                    double actOutputClassDistance = Math.abs(net.getOutputNeuron(j).getOutput() - classTargetValue);
                    for (int x = 0; x < net.getOutputNeuronCount(); x++) {
                        double otherClassDistance = Math.abs(net.getOutputNeuron(x).getOutput() - classTargetValue);
                        if (x != j && (actOutputClassDistance > otherClassDistance)) {
                            testsSuccessful[i] = false;
                            break;
                        }
                    }
                }
            }
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


    public static void main(String[] args) {
        double classTargetValue = 0.8D;
        double nonClassTargetValue = 0.1D;
        File mnistANN = new File("mnistANN_784i_6h_10o.xnet");
        int steps = 5;
        int epochs = 2;
        int nrHiddenLayers = 6;

        TT_TestResults testResults = neuralNetRun(mnistANN, classTargetValue, nonClassTargetValue, steps, epochs, nrHiddenLayers);
        System.out.println(testResults);
    }

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
