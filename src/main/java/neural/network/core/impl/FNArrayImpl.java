package neural.network.core.impl;

import java.util.Arrays;

import neural.network.core.FNArray;

public class FNArrayImpl implements FNArray {

    private Object object;

    private int[] shape;

    private int size;

    public FNArrayImpl(int[] shape, Object object) {
        this.shape = shape;
        this.object = object;
        this.size = Arrays.stream(shape).reduce(1, (r, s) -> r * s);
    }
    
    @Override
    public Object getObject() {
        return object;
    }

    @Override
    public float get(int... coord) {
        switch (getDimension()) {
            case 1:
                return ((float[]) object)[coord[0]];
            case 2:
                return ((float[][]) object)[coord[0]][coord[1]];
            case 3:
                return ((float[][][]) object)[coord[0]][coord[1]][coord[2]];
            default:
                return ((float[][][][]) object)[coord[0]][coord[1]][coord[2]][coord[3]];
        }
    }

    @Override
    public void set(float value, int... coord) {
        switch (getDimension()) {
            case 1:
                ((float[]) object)[coord[0]] = value;
                break;
            case 2:
                ((float[][]) object)[coord[0]][coord[1]] = value;
                break;
            case 3:
                ((float[][][]) object)[coord[0]][coord[1]][coord[2]] = value;
                break;
            default:
                ((float[][][][]) object)[coord[0]][coord[1]][coord[2]][coord[3]] = value;
                break;
        }
    }

    @Override
    public int getDimension() {
        return shape.length;
    }

    @Override
    public int[] getShape() {
        return Arrays.copyOf(shape, getDimension());
    }

    public int getShape(int i) {
        return shape[i];
    }

    @Override
    public int getSize() {
        return size;
    }

    @Override
    public String toString() {
        String s;
        switch (getDimension()) {
            case 1:
                s = Arrays.toString(((float[]) object));
                break;
            case 2:
                s = Arrays.deepToString(((float[][]) object));
                break;
            case 3:
                s = Arrays.deepToString(((float[][][]) object));
                break;
            default:
                s = Arrays.deepToString(((float[][][][]) object));
                break;
            }
        return "{object=" + s + ", shape=" + Arrays.toString(shape) + "}";
    }

}
