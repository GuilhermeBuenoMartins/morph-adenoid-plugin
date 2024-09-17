package neural.network.impl;

import java.util.stream.IntStream;

import neural.network.Activation;
import neural.network.Layer;
import neural.network.core.FNArray;
import neural.network.core.FNArrayFactory;

public class Dense implements Layer {

    private FNArray w;

    private FNArray b;

    private Activation activation;

    public Dense() {
    }

    public Dense(Activation activation) {
        this.activation = activation;
    }

    @Override
    public void load(FNArray w, FNArray b) {
        this.w = w;
        this.b = b;
    }

    @Override
    public FNArray predict(FNArray x) {
        switch (activation) {
            case TANH:
                return tanh(x);
            default:
                return sigmoid(x);
        }
    }

    private FNArray tanh(FNArray x) {
        FNArray y = FNArrayFactory.create(new int[] { x.getShape(0), w.getShape(1) }, 0);
        IntStream.range(0, x.getShape(0)).parallel().forEach(i -> {
            IntStream.range(0, w.getShape(1)).parallel().forEach(j -> {
                float[] multi = new float[x.getShape(1)];
                IntStream.range(0, x.getShape(1)).parallel().forEach(k -> {
                    multi[k] = x.get(i, k) * w.get(k, j);
                });
                double z = IntStream.range(0, x.getShape(1)).mapToDouble(k -> (double) multi[k]).sum() + b.get(j);
                double expZ = Math.exp(z);
                double expMinusZ = Math.exp(-z);
                float hx = (float) ((expZ - expMinusZ) / (expZ + expMinusZ));
                y.set(hx, i, j);
            });
        });
        return y;
    }

    private FNArray sigmoid(FNArray x) {
        FNArray y = FNArrayFactory.create(new int[] { x.getShape(0), w.getShape(1) }, 0);
        IntStream.range(0, x.getShape(0)).parallel().forEach(i -> {
            IntStream.range(0, w.getShape(1)).parallel().forEach(j -> {
                float[] multi = new float[x.getShape(1)];
                IntStream.range(0, x.getShape(1)).parallel().forEach(k -> {
                    multi[k] = x.get(i, k) * w.get(k, j);
                });
                double z = IntStream.range(0, x.getShape(1)).mapToDouble(k -> (double) multi[k]).sum() + b.get(j);
                float hx = (float) (1 / (1 + Math.exp(-z)));
                y.set(hx, i, j);
            });
        });
        return y;
    }

}
