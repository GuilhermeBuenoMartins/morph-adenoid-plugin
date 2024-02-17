package nn.layers;

import nn.Layer;
import nn.enums.Padding;
import nn.utils.Dilation;
import nn.utils.Output;

public class Dilation2D extends Layer {
    
    private float[][][][] w;

    private Padding padding = Padding.VALID;

    private int[] strides = {1, 1};

    public Dilation2D() {}

    public Dilation2D(float[][][][] w) {
        this.w = w;
    }

    public Dilation2D(Padding padding) {
        this.padding = padding;
    }
    
    public Dilation2D(float[][][][] w, Padding padding) {
        this.w = w;
        this.padding = padding;
    }

    public void loadWeights(float[][][][] w) {
        this.w = w;
    }

    private synchronized void dilation(float[][][][] y, float[][][] x, int yId, int yId1, int yId2) {
        for (int i = 0; i < w[0][0][0].length; i++) {
            y[yId][yId1][yId2][i] = Dilation.operation(x, w, new int[]{yId1 * strides[0], yId2 * strides[1]}, i);
        }
    }

    private synchronized void iterateColumns(float[][][][] y, float [][][] x, int yId, int yId1) {
        for (int i = 0; i < y[0][0].length; i++) {
            dilation(y, x, yId, yId1, i);
        }
    }

    private synchronized void iterateRows(float[][][][] y, float[][][] x, int yId) {
        for (int i = 0; i < y[0].length; i++) {
            iterateColumns(y, x, yId, i);
        }
    }

    private void iterateSamples(float[][][][] y, float[][][][] x) {
        for (int i = 0; i < y.length; i++) {
            iterateRows(y, x[i], i);
        }
    }

    public float[][][][] predict(float[][][][] x) {
        if (padding.equals(Padding.SAME)) { x = nn.utils.Padding.create(x, w, Float.NEGATIVE_INFINITY); }
        float[][][][] y = Output.create(x, w, strides);
        iterateSamples(y, x);
        return y;
    }
}
