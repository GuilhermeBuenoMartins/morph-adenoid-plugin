package nn;

import ch.systemsx.cisd.hdf5.IHDF5SimpleReader;
import nn.layers.Dense;
import nn.layers.Flatten;
import nn.layers.Opening2D;
import nn.layers.TopHatOpening2D;
import nn.utils.Basics;

public class Model {
    
    private Layer[] layers;

    public Model(Layer... layers) {
        this.layers = layers;
    }

    public void loadFrom(IHDF5SimpleReader reader, int layerId, int[] wShape) {
        if (layers[layerId] instanceof Dense) {
            ((Dense) layers[layerId]).loadWeights(reader.readFloatMatrix("weights"), reader.readFloatArray("bias"));
        } else if (layers[layerId] instanceof TopHatOpening2D) {
            ((TopHatOpening2D) layers[layerId]).loadWeights(
                Basics.create(new float[wShape[0]][wShape[1]][wShape[2]][wShape[3]], reader.readFloatArray("weights")));
        } else {
            ((Opening2D) layers[layerId]).loadWeights(
                Basics.create(new float[wShape[0]][wShape[1]][wShape[2]][wShape[3]], reader.readFloatArray("weights")));
        }
    }
    
    public float[][] predict(float[][][][] x) {
        float[][][][] y = null;
        float[][] x2D = null;
        float[][] y2D = null;
        for (Layer layer : layers) {
            if (layer instanceof TopHatOpening2D) {
                y = ((TopHatOpening2D) layer).predict(x);
                x = y;
            } else if (layer instanceof Opening2D) {
                y = ((Opening2D) layer).predict(x);
                x = y;
            } else if (layer instanceof Dense){
                y2D = ((Dense) layer).predict(x2D);
                x2D = y2D;
            } else {
                y = null;
                y2D = ((Flatten) layer).predict(x);
                x = null;
                x2D = y2D;
            }
        }
        return y2D;
    }
}
