package neural.network.core;

import java.util.stream.IntStream;

import neural.network.core.impl.F1Array;
import neural.network.core.impl.F2Array;
import neural.network.core.impl.F3Array;
import neural.network.core.impl.F4Array;

public interface FNArrayFactory {

    public static FNArray create(int[] shape, float value) {
        Verify.exceededDimension(4, shape);
        int size = IntStream.range(0, shape.length).reduce(1, (r, i) -> r * shape[i]);
        float[] values = new float[size];
        IntStream.range(0, size).parallel().forEach(i -> values[i] = value);
        switch (shape.length) {
            case 1:
                return new F1Array(shape, values);
            case 2:
                return new F2Array(shape, values);
            case 3:
                return new F3Array(shape, values);
            default:
                return new F4Array(shape, values);
        }
    }

    public static FNArray create(int[] shape, float... values) {
        Verify.exceededDimension(4, shape);
        Verify.valuesLength(values.length, shape);
        switch (shape.length) {
            case 1:
                return new F1Array(shape, values);
            case 2:
                return new F2Array(shape, values);
            case 3:
                return new F3Array(shape, values);
            default:
                return new F4Array(shape, values);
        }
    }

}
