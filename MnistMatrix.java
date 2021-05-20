import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class MnistMatrix {

    private int [][] data;

    private int nRows;
    private int nCols;

    private int label;

    public MnistMatrix(int nRows, int nCols) {
        this.nRows = nRows;
        this.nCols = nCols;

        data = new int[nRows][nCols];
    }

    public int getValue(int r, int c) {
        return data[r][c];
    }

    public void setValue(int row, int col, int value) {
        data[row][col] = value;
    }

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }

    public int getNumberOfRows() {
        return nRows;
    }

    public int getNumberOfColumns() {
        return nCols;
    }

    public List<Integer> getDataAsList() {
        var tmp = new ArrayList<Integer>(this.data.length * this.data[0].length);
        for (int[] i : this.data) {
            for (int j : i) {
                tmp.add(j);
            }
        }
        return tmp;
    }

}
