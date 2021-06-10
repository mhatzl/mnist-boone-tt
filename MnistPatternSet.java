import java.util.ArrayList;
import java.util.List;

public class MnistPatternSet {

    private final List<Double> inputs;
    private final List<Double> targets;

    public MnistPatternSet(MnistMatrix mnistMatrix, double[] classTargetValues, double[] nonClassTargetValues) {
        this.targets = new ArrayList<>(10);
        this.inputs = new ArrayList<>(784);

        int label = mnistMatrix.getLabel();
        for (int i = 0; i < classTargetValues.length; i++) {
            if (i != label) {
                this.targets.add(nonClassTargetValues[i]);
            } else {
                this.targets.add(classTargetValues[i]);
            }
        }

        for (int i = 0; i < mnistMatrix.getNumberOfRows(); i++) {
            for (int j = 0; j < mnistMatrix.getNumberOfColumns(); j++) {
                this.inputs.add((double)mnistMatrix.getValue(i,j));
            }
        }
    }

    public List<Double> getInputs() { return this.inputs; }

    public List<Double> getTargets() { return this.targets; }

}
