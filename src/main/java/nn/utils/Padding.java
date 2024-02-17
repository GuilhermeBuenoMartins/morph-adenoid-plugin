package nn.utils;

public class Padding {
    
    private synchronized static void iterateColumns(float[][] xPad, float[][] x, int pad) {
        for (int i = 0; i < x.length; i++) {
            System.arraycopy(x[i], 0, xPad[i + pad], 0, x[i].length);
        }
    }

    private synchronized static void iterateRows(float[][][] xPad, float[][][] x, int pad, int pad1) {
        for (int i = 0; i < x.length; i++) {
            iterateColumns(xPad[i + pad], x[i], pad1);
        }
    }

    private synchronized static void iterateSamples(float[][][][] xPad, float[][][][] x, int pad, int pad1) {
        for (int i = 0; i < x.length; i++) {
            iterateRows(xPad[i], x[i], pad, pad1);
        }
    }

    public static float[][][][] create(float[][][][] x, float[][][][] w, float value) {
        float[][][][] xPad = Basics.create(
            new float[x.length][x[0].length + w.length - 1][x[0][0].length + w[0].length -1][x[0][0][0].length], value);
        iterateSamples(xPad, x, (w.length - 1)/ 2, (w[0].length - 1)/ 2);
        return xPad;
    }
}
