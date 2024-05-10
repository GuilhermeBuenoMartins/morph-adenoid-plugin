package config;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Properties;

import ch.systemsx.cisd.hdf5.HDF5Factory;
import ij.IJ;
import neural.network.Activation;
import neural.network.Model;
import neural.network.impl.Dense;
import neural.network.impl.Flatten;
import neural.network.impl.Opening2D;
import neural.network.impl.TopHatOpening2D;

public class Config {

    public static final String SEP = ",";

    private static final String SETTINGS_FILE = "settings.properties";

    private static ClassLoader loader = Config.class.getClassLoader();

    private static Properties properties = null;

    private static Model model = null;

    private Config() {
    }

    public static final Properties getProperties() {
        if (properties != null) {
            return properties;
        }
        try {
            properties = new Properties();
            properties.load(loader.getResourceAsStream(SETTINGS_FILE));
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e.getMessage());
        } catch (IOException e) {
            throw new RuntimeException(e.getMessage());
        }
        return properties;
    }

    public static Model getModel() {
        if (model == null) { model = loadModel(); }
        return model;
    }

    private static Model loadModel() {
        model = buildModel();
        final Path DIR = Path.of(IJ.getDir("plugins"), getProperties().getProperty("layers.subdirectory"));
        final String[] layerIds = getProperties().getProperty("layers.ids").split(SEP);
        for (String layerId : layerIds) {
            int[] wShape = Arrays.stream(
                    getProperties().getProperty(String.format("layers.layer_%s.shape", layerId)).split(SEP))
                    .mapToInt(s -> Integer.parseInt(s)).toArray();
            model.loadLayer(Integer.parseInt(layerId), wShape,
                    HDF5Factory.openForReading(String.format("%s/layer_%s.h5", DIR.toAbsolutePath(), layerId)));
        }
        return model;
    }

    private static Model buildModel() {
        return new Model(new TopHatOpening2D(), new Opening2D(), new Flatten(),
                new Dense(Activation.TANH), new Dense(Activation.TANH), new Dense(Activation.SIGMOID));
    }
}