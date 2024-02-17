package nn.layers;

import nn.Layer;
import nn.enums.Activation;

public class Dense extends Layer {
    
    private float[][] w;

    private float[] b;

    private Activation activation = Activation.TANH;
    
    public Dense() {}

    public Dense(Activation activation) {
        this.activation = activation;
    }

    public Dense(float[][] w, float[] b) {
        this.w = w;
        this.b = b;
    }

    public Dense(float[][] w, float[] b, Activation activation) {
        this.w = w;
        this.b = b;
        this.activation = activation;
    }

    public void loadWeights(float[][] w, float[] b) {
        this.w = w;
        this.b = b;
    }

    private synchronized void tanh(float[] y, float[] x) {
        for (int i = 0; i < w[0].length; i++) {
            float sum = 0;
            for (int j = 0; j < x.length; j++) {
                sum += x[j] * w[j][i];
            }
            float z = sum + b[i];
            float e_z = (float) Math.exp(z);
            float e__z = (float) Math.exp(-z);
            y[i] = (e_z - e__z) / (e_z + e__z);
        }
    }


    private void iterateTanHSamples(float[][] y, float[][] x) {
        for (int i = 0; i < x.length; i++) {
            tanh(y[i], x[i]);
        }
    }

    private void sigmoid(float[]y, float[] x) {
        for (int i = 0; i < w[0].length; i++) {
            float sum = 0;
            for (int j = 0; j < x.length; j++) {
                sum += x[j] * w[j][i];
            }
            float z = sum + b[i];
            float e__z = (float) Math.exp(-z);
            y[i] = 1 / (1 + e__z);
        }
    }

    private void iterateSigmoidSamples(float[][] y, float[][] x) {
        for (int i = 0; i < x.length; i++) {
            sigmoid(y[i], x[i]);
        }
    }

    public float[][] predict(float[][] x) {
        float[][] y = new float[x.length][w[0].length];
        if (activation.equals(Activation.TANH)) {
            iterateTanHSamples(y, x);
        } else {
            iterateSigmoidSamples(y, x);
        }
        return y;
    }
}
