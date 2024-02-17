package nn.utils;

import java.util.Arrays;

public class Basics {
    
    public synchronized static float[][] create(float[][] array, float... nums) {
        for (int i = 0; i < array.length; i++) {
            System.arraycopy(nums, i * array[0].length, array[i], 0, array[0].length);
        }
        return array;
    }

    public synchronized static float[][][] create(float[][][] array, float... nums) {
        for (int i = 0; i < array.length; i++) {
            int from = i * array[0].length * array[0][0].length;
            int to = (i + 1) * array[0].length * array[0][0].length;
            create(array[i], Arrays.copyOfRange(nums, from, to));
        }
        return array;
    }

    public synchronized static float[][][][] create(float[][][][] array, float... nums) {
        for (int i = 0; i < array.length; i++) {
            int from = i * array[0].length * array[0][0].length * array[0][0][0].length;
            int to = (i + 1) * array[0].length * array[0][0].length * array[0][0][0].length;
            create(array[i], Arrays.copyOfRange(nums, from, to));
        }
        return array;
    }

    public synchronized static float[] create(float[] array, float num) {
        for (int i = 0; i < array.length; i++) {
            array[i] = num;
        }
        return array;
    }

    public synchronized static float[][] create(float[][] array, float num) {
        for (int i = 0; i < array.length; i++) {
            create(array[i], num);
        }
        return array;
    }

    public synchronized static float[][][] create(float[][][] array, float num) {
        for (int i = 0; i < array.length; i++) {
            create(array[i], num);
        }
        return array;
    }

    public synchronized static float[][][][] create(float[][][][] array, float num) {
        for (int i = 0; i < array.length; i++) {
            create(array[i], num);
        }
        return array;
    }

    public static int[] shape(float[][] array) {
        return new int[]{array.length, array[0].length};
    }

    public static int[] shape(float[][][] array) {
        return new int[]{array.length, array[0].length, array[0][0].length};
    }

    public static int[] shape(float[][][][] array) {
        return new int[]{array.length, array[0].length, array[0][0].length, array[0][0][0].length};
    }

    public synchronized static float[][] copy(float[][] array) {
        int[] arrayShape = shape(array);
        float[][] copy = new float[arrayShape[0]][arrayShape[1]];
        for (int i = 0; i < array.length; i++) {
            System.arraycopy(array[i], 0, copy[i], 0, array[i].length);
        }
        return copy;
    }

    public synchronized static float[][][] copy(float[][][] array) {
        int[] arrayShape = shape(array);
        float[][][] copy = new float[arrayShape[0]][arrayShape[1]][arrayShape[2]];
        for (int i = 0; i < array.length; i++) {
            copy[i] = copy(array[i]);
        }
        return copy;
    }

    public synchronized static float[][][][] copy(float[][][][] array) {
        int[] arrayShape = shape(array);
        float[][][][] copy = new float[arrayShape[0]][arrayShape[1]][arrayShape[2]][arrayShape[3]];
        for (int i = 0; i < array.length; i++) {
            copy[i] = copy(array[i]);
        }
        return copy;
    }
}
