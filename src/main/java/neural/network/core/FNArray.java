package neural.network.core;

public interface FNArray {

    Object getObject();

    float get(int... i);

    void set(float value, int... coord);

    int getDimension();

    int[] getShape();

    int getShape(int i);

    int getSize();

    @Override
    String toString();

}
