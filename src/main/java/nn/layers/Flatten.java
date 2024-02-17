package nn.layers;

import java.util.Arrays;

import nn.Layer;
import nn.utils.Basics;

public class Flatten extends Layer {

    private synchronized void flat(float[] y, float[][][] x) {
        int yId = 0;
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                for (int k = 0; k < x[0][0].length; k++) {
                    y[yId] = x[i][j][k];
                    yId++;
                }
            }
        }
    }

    private void iterateSamples(float[][] y, float[][][][] x) {
        for (int i = 0; i < x.length; i++) {
            flat(y[i], x[i]);
        }
    }

    public float[][] predict(float[][][][] x) {
        int[] xShape = Basics.shape(x);
        float[][] y = new float[xShape[0]][Arrays.stream(xShape, 1, xShape.length).reduce(1, (r, s) -> r * s)];
        iterateSamples(y, x);
        return y;
    }
    
}
