package neural.network.core;

public interface FNArray {

    float get(int... coord);

    void set(float value, int... coord);

    int getDimension();

    int[] getShape();

    int getShape(int i);

    int getSize();

    @Override
    String toString();

}
