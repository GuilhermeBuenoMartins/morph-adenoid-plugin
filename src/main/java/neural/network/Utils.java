package neural.network;

import java.util.stream.IntStream;

import neural.network.core.FNArray;
import neural.network.core.FNArrayFactory;

public interface Utils {

    public static FNArray padding(FNArray x, int wH, int wW, float value) {
        FNArray xPad = FNArrayFactory.create(
                new int[] { x.getShape(0), x.getShape(1) + wH - 1, x.getShape(2) + wW - 1, x.getShape(3) }, value);
        int padH = (wH - 1) / 2;
        int padW = (wW - 1) / 2;
        IntStream.range(0, x.getShape(0)).parallel().forEach(i -> {
            IntStream.range(0, x.getShape(1)).parallel().forEach(j -> {
                IntStream.range(0, x.getShape(2)).parallel().forEach(k -> {
                    IntStream.range(0, x.getShape(3)).parallel().forEach(l -> {
                        xPad.set(x.get(i, j, k, l), i, j + padH, k + padW, l);
                    });
                });
            });
        });
        return xPad;
    }

    public static FNArray getY(int numSamples, int xH, int xW, int wH, int wW, int[] strides, int outCh) {
        return FNArrayFactory
                .create(new int[] { numSamples, 1 + (xH - wH) / strides[0], 1 + (xW - wW) / strides[1], outCh }, 0);
    }

    public static float erosion(FNArray x, int[] xPoint, FNArray w, int wCh, int wId) {
        float y = Float.POSITIVE_INFINITY;
        for (int i = 0; i < w.getShape(0); i++) {
            for (int j = 0; j < w.getShape(1); j++) {
                y = Math.min(y,
                        x.get(xPoint[0], xPoint[1] + i, xPoint[2] + j, xPoint[3])
                                - w.get(w.getShape(0) - 1 - i, w.getShape(1) - 1 - j, wCh, wId));
            }
        }
        return y;
    }

    public static float dilation(FNArray x, int[] xPoint, FNArray w, int wCh, int wId) {
        float y = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < w.getShape(0); i++) {
            for (int j = 0; j < w.getShape(1); j++) {
                y = Math.max(y,
                        x.get(xPoint[0], xPoint[1] + i, xPoint[2] + j, xPoint[3])
                                + w.get(i, j, wCh, wId));
            }
        }
        return y;
    }

}
