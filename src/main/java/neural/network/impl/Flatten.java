package neural.network.impl;

import java.util.stream.IntStream;

import neural.network.Layer;
import neural.network.core.FNArray;
import neural.network.core.FNArrayFactory;

public class Flatten implements Layer {

    @Override
    public void load(FNArray w, FNArray b) {
    }

    @Override
    public FNArray predict(FNArray x) {
        return flatten(x);
    }

    private FNArray flatten(FNArray x) {
        float[] values = new float[x.getSize()];
        IntStream.range(0, x.getShape(0)).parallel().forEach(i -> {
            IntStream.range(0, x.getShape(1)).parallel().forEach(j -> {
                IntStream.range(0, x.getShape(2)).parallel().forEach(k -> {
                    IntStream.range(0, x.getShape(3)).parallel().forEach(l -> {
                        values[x.getShape(3) * (x.getShape(2) * (x.getShape(1) * i + j) + k) + l] = x.get(i, j, k, l);
                    });
                });
            });
        });
        return FNArrayFactory.create(new int[] { x.getShape(0), x.getSize() / x.getShape(0) }, values);
    }

}
