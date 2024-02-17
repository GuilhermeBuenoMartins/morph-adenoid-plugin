package nn.layers;

import nn.Layer;
import nn.enums.Padding;
import nn.utils.Dilation;
import nn.utils.Erosion;
import nn.utils.Output;

public class Opening2D extends Layer {

    private float[][][][] w;

    private Padding padding = Padding.VALID;

    private int[] strides = { 1, 1 };

    public Opening2D() {
    }

    public Opening2D(float[][][][] w) {
        this.w = w;
    }

    public Opening2D(Padding padding) {
        this.padding = padding;
    }

    public Opening2D(float[][][][] w, Padding padding) {
        this.w = w;
        this.padding = padding;
    }

    public void loadWeights(float[][][][] w) {
        this.w = w;
    }

    private synchronized void erosion(float[][][][] y, float[][][]x, int yId, int yId1, int yId2, int ch, int wId) {
        y[yId][yId1][yId2][ch + w[0][0].length * wId] = Erosion.operation(
            x, w, new int[]{yId1 * strides[0], yId2 * strides[1]}, ch, wId);
    }

    private synchronized void iterateErosionColumns(float[][][][] y, float [][][] x, int yId, int yId1, int ch, int wId) {
        for (int i = 0; i < y[0][0].length; i++) {
            erosion(y, x, yId, yId1, i, ch, wId);
        }
    }

    private synchronized void iterateErosionRows(float[][][][] y, float[][][] x, int yId, int ch, int wId) {
        for (int i = 0; i < y[0].length; i++) {
            iterateErosionColumns(y, x, yId, i, ch, wId);
        }
    }

    private void iterateErosionSamples(float[][][][] y, float[][][][] x, int ch, int wId) {
        for (int i = 0; i < y.length; i++) {
            iterateErosionRows(y, x[i], i, ch, wId);
        }
    }

    private synchronized void iterateErosionChannels(float[][][][] y, float[][][][] x, int wId) {
        for (int i = 0; i < w[0][0].length; i++) {
            iterateErosionSamples(y, x, i, wId);
        }
    }

    private synchronized void iterateErosionFilters(float[][][][] y, float[][][][] x) {
        for (int i = 0; i < w[0][0][0].length; i++) {
            iterateErosionChannels(y, x, i);
        }
    }

    private synchronized void dilation(float[][][][] y, float[][][]x, int yId, int yId1, int yId2, int xCh, int wCh, int wId) {
        y[yId][yId1][yId2][xCh] = Dilation.operation(
            x, w, new int[]{yId1 * strides[0], yId2 * strides[1]}, xCh, wCh, wId);
    }

    private synchronized void iterateDilationColumns(float[][][][] y, float [][][] x, int yId, int yId1, int xCh, int wCh, int wId) {
        for (int i = 0; i < y[0][0].length; i++) {
            dilation(y, x, yId, yId1, i, xCh, wCh,wId);
        }
    }

    private synchronized void iterateDilationRows(float[][][][] y, float[][][] x, int yId,  int xCh, int wCh, int wId) {
        for (int i = 0; i < y[0].length; i++) {
            iterateDilationColumns(y, x, yId, i, xCh, wCh, wId);
        }
    }

    private void iterateDilationSamples(float[][][][] y, float[][][][] x,  int xCh, int wCh, int wId) {
        for (int i = 0; i < y.length; i++) {
            iterateDilationRows(y, x[i], i, xCh, wCh, wId);
        }
    }

    private synchronized void iterateDilationChannels(float[][][][] y, float[][][][] x, int xCh, int wId) {
        for (int i = 0; i < w[0][0].length; i++) {
            iterateDilationSamples(y, x, xCh, i, wId);
            xCh++;
        }
    }

    private synchronized void iterateDilationFilters(float[][][][] y, float[][][][] x) {
        int xCh = 0;
        for (int i = 0; i < w[0][0][0].length; i++) {
            iterateDilationChannels(y, x, xCh, i);
            xCh += w[0][0].length;
        }
    }

    public float[][][][] predict(float[][][][] x) {
        if (padding.equals(Padding.SAME)) {
            x = nn.utils.Padding.create(x, w, Float.POSITIVE_INFINITY);
        }
        int ch = x[0][0][0].length * w[0][0][0].length;
        float[][][][] y = Output.custom(x, w, strides, ch);
        iterateErosionFilters(y, x);
        x = y;
        if (padding.equals(Padding.SAME)) {
            x = nn.utils.Padding.create(x, w, Float.NEGATIVE_INFINITY);
        }
        y = Output.custom(x, w, strides, ch);
        iterateDilationFilters(y, x);
        return y;
    }

}
