package neural.network.core.impl;

import java.util.Arrays;
import neural.network.core.FNArray;

public class F2Array implements FNArray {

    private int[] shape;
    
    private float[] values;

    public F2Array(int[] shape, float[] values) {
        this.shape = shape;
        this.values = values;
    }

    @Override
    public float get(int... coord) {
        return values[shape[1] * coord[0] + coord[1]];
    }

    @Override
    public int getDimension() {
        return 2;
    }

    @Override
    public int[] getShape() {
        return Arrays.copyOf(shape, shape.length);
    }

    @Override
    public int getShape(int i) {
        return shape[i];
    }

    @Override
    public int getSize() {
        return values.length;
    }

    @Override
    public void set(float value, int... coord) {
        values[shape[1] * coord[0] + coord[1]] = value;
    }

}
