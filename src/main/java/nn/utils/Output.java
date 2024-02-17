package nn.utils;

public class Output {
    
    public static float[][][][] create(float[][][][] x, float[][][][] w, int[] strides) {
        return new float[x.length]
            [Math.floorDiv(x[0].length - w.length, strides[0]) + 1]
            [Math.floorDiv(x[0][0].length - w[0].length, strides[1]) + 1]
            [w[0][0][0].length];
    }

    public static float[][][][] custom(float[][][][] x, float[][][][] w, int[] strides, int ch) {
        return new float[x.length]
            [Math.floorDiv(x[0].length - w.length, strides[0]) + 1]
            [Math.floorDiv(x[0][0].length - w[0].length, strides[1]) + 1]
            [ch];
    }

}
