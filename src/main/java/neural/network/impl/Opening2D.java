package neural.network.impl;

import java.util.stream.IntStream;

import neural.network.Layer;
import neural.network.Utils;
import neural.network.core.FNArray;
import neural.network.core.FNArrayFactory;

public class Opening2D implements Layer {

    private FNArray w;

    private int[] strides = new int[] { 1, 1 };

    private boolean padding = false;

    public Opening2D() {
    }

    public Opening2D(int[] strides, boolean padding) {
        this.strides = strides;
        this.padding = padding;
    }

    public Opening2D(boolean padding) {
        this.padding = padding;
    }

    @Override
    public void load(FNArray w, FNArray b) {
        this.w = w;
    }

    @Override
    public FNArray predict(FNArray x) {
        return opening2D(x);
    }

    public FNArray opening2D(FNArray x) {
        FNArray eX = padding? Utils.padding(x, w.getShape(0), w.getShape(1), Float.POSITIVE_INFINITY): x;
        FNArray[] openings = new FNArray[w.getShape(3) * eX.getShape(3)];
        IntStream.range(0, w.getShape(3)).parallel().forEach(i -> {
            IntStream.range(0, eX.getShape(3)).parallel().forEach(j -> {
                openings[i * eX.getShape(3) + j] = dilation2D(erosion2D(eX, j, i), j, i);
            });
        });
        FNArray y = FNArrayFactory.create(
            new int[] {openings[0].getShape(0), openings[0].getShape(1), openings[0].getShape(2), openings.length}, 0);
        IntStream.range(0, openings[0].getShape(0)).parallel().forEach(i -> {
            IntStream.range(0, openings[0].getShape(1)).parallel().forEach(j -> {
                IntStream.range(0, openings[0].getShape(2)).parallel().forEach(k -> {
                    IntStream.range(0, openings.length).parallel().forEach(l -> {
                        y.set(openings[l].get(i, j, k, 0), i, j, k, l);
                    });
                });
            });
        });
        return y;
    }

    public FNArray erosion2D(FNArray x, int ch, int wId) {
        FNArray y = Utils.getY(x.getShape(0), x.getShape(1), x.getShape(2), w.getShape(0), w.getShape(1), strides, 1);
        IntStream.range(0, y.getShape(0)).parallel().forEach(i -> {
            IntStream.range(0, y.getShape(1)).parallel().forEach(j -> {
                IntStream.range(0, y.getShape(2)).parallel().forEach(k -> {
                    IntStream.range(0, y.getShape(3)).parallel().forEach(l -> {
                        y.set(Utils.erosion(x, new int[] {i, j * strides[0], k * strides[1], ch}, w, ch, wId), i, j, k, l);
                    });
                });
            });
        });
        return y;
    }

    public FNArray dilation2D(FNArray x, int ch, int wId) {
        FNArray dX = padding? Utils.padding(x, w.getShape(0), w.getShape(1), Float.NEGATIVE_INFINITY): x;
        FNArray y = Utils.getY(dX.getShape(0), dX.getShape(1), dX.getShape(2), w.getShape(0), w.getShape(1), strides, 1);
        IntStream.range(0, y.getShape(0)).parallel().forEach(i -> {
            IntStream.range(0, y.getShape(1)).parallel().forEach(j -> {
                IntStream.range(0, y.getShape(2)).parallel().forEach(k -> {
                    IntStream.range(0, y.getShape(3)).parallel().forEach(l -> {
                        y.set(Utils.dilation(dX, new int[] {i, j * strides[0], k * strides[1], 0}, w, ch, wId), i, j, k, l);
                    });
                });
            });
        });
        return y;
    }

}
