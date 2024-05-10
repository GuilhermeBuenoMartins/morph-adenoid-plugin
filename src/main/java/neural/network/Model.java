package neural.network;

import ch.systemsx.cisd.hdf5.IHDF5SimpleReader;
import neural.network.core.FNArray;
import neural.network.core.FNArrayFactory;
import neural.network.impl.Dense;

public class Model {

    private Layer[] layers;

    public Model(Layer... layers) {
        this.layers = layers;
    }

    public void loadLayer(int layerId, int[] wShape, IHDF5SimpleReader reader) {
        if (layers[layerId] instanceof Dense) {
            FNArray w = FNArrayFactory.create(wShape, reader.readFloatArray("weights"));
            FNArray b = FNArrayFactory.create(new int[] { wShape[1]}, reader.readFloatArray("bias"));
            layers[layerId].load(w, b);
        } else {
            FNArray w = FNArrayFactory.create(wShape, reader.readFloatArray("weights"));
            layers[layerId].load(w, null);
        }
    }

    public FNArray predict(FNArray x) {
        for (Layer layer: layers) {
            x = layer.predict(x);
        }
        return x;
    }

}
