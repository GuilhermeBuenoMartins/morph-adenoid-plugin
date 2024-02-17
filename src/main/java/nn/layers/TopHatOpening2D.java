package nn.layers;

import nn.Layer;
import nn.enums.Padding;
import nn.utils.Basics;

public class TopHatOpening2D extends Layer {

    private float[][][][] w;

    private Padding padding = Padding.SAME;

    public TopHatOpening2D() {
    }

    public TopHatOpening2D(float[][][][] w) {
        this.w = w;
    }

    public TopHatOpening2D(Padding padding) {
        this.padding = padding;
    }

    public void loadWeights(float[][][][] w) {
        this.w = w;
    }

    private synchronized void topHatOpening(float[] y, float[] yOpening, float[] x, int yId) {
        for (int i = 0; i < x.length; i++) {
            y[yId + i] = x[i] - yOpening[yId + i];
        }
    }

    private synchronized void iterateColumns(float[][] y, float[][] yOpening, float[][] x, int yId) {
        for (int i = 0; i < y.length; i++) {
            topHatOpening(y[i], yOpening[i], x[i], yId);
        }
    }

    private synchronized void iterateRows(float[][][] y, float[][][] yOpening, float[][][] x, int yId) {
        for (int i = 0; i < y.length; i++) {
            iterateColumns(y[i], yOpening[i], x[i], yId);
        }
    }

    private synchronized void iterateSamples(float[][][][] y, float[][][][] yOpening, float[][][][] x, int yId) {
        for (int i = 0; i < y.length; i++) {
            iterateRows(y[i], yOpening[i], x[i], yId);
        }
    }

    private synchronized void iterateChannels(float[][][][] y, float[][][][] yOpening, float[][][][] x) {
        for (int i = 0; i < y[0][0][0].length; i += x[0][0][0].length) {
            iterateSamples(y, yOpening, x, i);
        }
    }

    public float[][][][] predict(float[][][][] x) {
        float[][][][] yOpening = new Opening2D(w, padding).predict(x);
        float[][][][] y = Basics.copy(yOpening);
        iterateChannels(y, yOpening, x);
        return y;
    }
}
