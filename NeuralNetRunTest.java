import org.junit.jupiter.api.Test;
import java.util.Arrays;
import java.util.List;


class NeuralNetRunTest {

    @Test
    void testSuccessfulTest_Successful() {
        List<Double> targets = Arrays.asList(0.0, 1.0, 0.0);
        double[] outputs = new double[]{0.1, 0.8, 0.2};
        double[] classTargetValues = new double[]{1.0, 1.0, 1.0};

        boolean actTestSuccessful = Main.wasTestSuccessful(targets, outputs, classTargetValues);

        assert(actTestSuccessful);
    }

    @Test
    void testSuccessfulTest_Failed() {
        List<Double> targets = Arrays.asList(0.0, 1.0, 0.0);
        double[] outputs = new double[]{0.1, 0.8, 0.85};
        double[] classTargetValues = new double[]{1.0, 1.0, 1.0};

        boolean actTestSuccessful = Main.wasTestSuccessful(targets, outputs, classTargetValues);

        assert(!actTestSuccessful);
    }

    @Test
    void testSuccessfulTest_FailedMultiClasses() {
        List<Double> targets = Arrays.asList(0.0, 0.7, 0.0);
        double[] outputs = new double[]{0.1, 0.8, 0.85};
        double[] classTargetValues = new double[]{1.0, 0.7, 0.8};

        boolean actTestSuccessful = Main.wasTestSuccessful(targets, outputs, classTargetValues);

        assert(!actTestSuccessful);
    }

    @Test
    void testSuccessfulTest_SuccessfulMultiClasses() {
        List<Double> targets = Arrays.asList(0.0, 0.7, 0.0);
        double[] outputs = new double[]{0.1, 0.8, 0.67};
        double[] classTargetValues = new double[]{1.0, 0.7, 0.8};

        boolean actTestSuccessful = Main.wasTestSuccessful(targets, outputs, classTargetValues);

        assert(actTestSuccessful);
    }

}