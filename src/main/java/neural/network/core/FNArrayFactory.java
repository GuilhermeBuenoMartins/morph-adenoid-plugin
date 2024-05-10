package neural.network.core;

import java.util.stream.IntStream;

import neural.network.core.impl.FNArrayImpl;

public interface FNArrayFactory {

    public static FNArray create(int[] shape, float value) {
        Verify.exceededDimension(4, shape);
        switch (shape.length) {
            case 1:
                float[] values1D = new float[shape[0]];
                IntStream.range(0, shape[0]).parallel().forEach(i -> values1D[i] = value);
                return new FNArrayImpl(shape, values1D);
            case 2:
                float[][] values2D = new float[shape[0]][shape[1]];
                IntStream.range(0, shape[0]).parallel().forEach(i -> {
                    IntStream.range(0, shape[1]).parallel().forEach(j -> values2D[i][j] = value);
                });
                return new FNArrayImpl(shape, values2D);
            case 3:
                float[][][] values3D = new float[shape[0]][shape[1]][shape[2]];
                IntStream.range(0, shape[0]).parallel().forEach(i -> {
                    IntStream.range(0, shape[1]).parallel().forEach(j -> {
                        IntStream.range(0, shape[2]).parallel().forEach(k -> values3D[i][j][k] = value);
                    });
                });
                return new FNArrayImpl(shape, values3D);
            default:
                float[][][][] values4D = new float[shape[0]][shape[1]][shape[2]][shape[3]];
                IntStream.range(0, shape[0]).parallel().forEach(i -> {
                    IntStream.range(0, shape[1]).parallel().forEach(j -> {
                        IntStream.range(0, shape[2]).parallel().forEach(k -> {
                            IntStream.range(0, shape[3]).parallel().forEach(l -> values4D[i][j][k][l] = value);
                        });
                    });
                });
                return new FNArrayImpl(shape, values4D);
        }
    }

    public static FNArray create(int[] shape, float... values) {
        Verify.exceededDimension(4, shape);
        Verify.valuesLength(values.length, shape);
        switch (shape.length) {
            case 1:
                return new FNArrayImpl(shape, values);
            case 2:
                float[][] values2D = new float[shape[0]][shape[1]];
                IntStream.range(0, shape[0]).parallel().forEach(i -> {
                    IntStream.range(0, shape[1]).parallel().forEach(
                        j -> values2D[i][j] = values[i * shape[1] + j]);
                });
                return new FNArrayImpl(shape, values2D);
            case 3:
                float[][][] values3D = new float[shape[0]][shape[1]][shape[2]];
                IntStream.range(0, shape[0]).parallel().forEach(i -> {
                    IntStream.range(0, shape[1]).parallel().forEach(j -> {
                        IntStream.range(0, shape[2]).parallel().forEach(
                            k -> values3D[i][j][k] = values[shape[2] * (shape[1] * i + j) + k]);
                    });
                });
                return new FNArrayImpl(shape, values3D);
            default:
                float[][][][] values4D = new float[shape[0]][shape[1]][shape[2]][shape[3]];
                IntStream.range(0, shape[0]).parallel().forEach(i -> {
                    IntStream.range(0, shape[1]).parallel().forEach(j -> {
                        IntStream.range(0, shape[2]).parallel().forEach(k -> {
                            IntStream.range(0, shape[3]).parallel().forEach(
                                l -> values4D[i][j][k][l] = values[shape[3] * (shape[2] * (shape[1] * i + j) + k) + l]);
                        });
                    });
                });
                return new FNArrayImpl(shape, values4D);
        }
    }

    public static FNArray create(int[] shape, float[][] values) {
        Verify.exceededDimension(4, shape);
        Verify.shape(shape, values);
        return new FNArrayImpl(shape, values);
    }
    
    public static FNArray create(int[] shape, float[][][] values) {
        Verify.exceededDimension(4, shape);
        Verify.shape(shape, values);
        return new FNArrayImpl(shape, values);
    }
    
    public static FNArray create(int[] shape, float[][][][] values) {
        Verify.exceededDimension(4, shape);
        Verify.shape(shape, values);
        return new FNArrayImpl(shape, values);
    }

}
