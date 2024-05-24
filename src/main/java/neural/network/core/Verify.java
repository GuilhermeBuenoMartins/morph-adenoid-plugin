package neural.network.core;

import java.util.Arrays;

public interface Verify {

    public static void exceededDimension(int allow, int[] shape) {
        if (shape.length > allow) {
            throw new UnsupportedOperationException("This operation support until " + allow + " dimension(s).");
        }
    }

    public static void valuesLength(int length, int[] shape) {
        int size = Arrays.stream(shape).reduce(1, (r, s) -> r * s);
        if (length != size) {
            String s = "The shape " + Arrays.toString(shape) + " is not compatible with values length " + length + ".";
            throw new IllegalArgumentException(s);
        }
    }

    public static void shape(int[] shape, float[][] values) {
        if (shape[0] != values.length || shape[1] != values[0].length) {
            String s = "The shape " + Arrays.toString(shape) + " is not compatible with values";
            throw new IllegalArgumentException(s);
        }
    }

    public static void shape(int[] shape, float[][][] values) {
        if (shape[0] != values.length || shape[1] != values[0].length || shape[2] != values[0][0].length) {
            String s = "The shape " + Arrays.toString(shape) + " is not compatible with values";
            throw new IllegalArgumentException(s);
        }
    }

    public static void shape(int[] shape, float[][][][] values) {
        if (shape[0] != values.length || shape[1] != values[0].length || shape[2] != values[0][0].length
                || shape[3] != values[0][0][0].length) {
            String s = "The shape " + Arrays.toString(shape) + " is not compatible with values";
            throw new IllegalArgumentException(s);
        }
    }
}
