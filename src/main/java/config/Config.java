package config;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.Properties;

import ch.systemsx.cisd.hdf5.HDF5Factory;
import ij.IJ;
import nn.Model;
import nn.enums.Activation;
import nn.enums.Padding;
import nn.layers.Dense;
import nn.layers.Flatten;
import nn.layers.Opening2D;
import nn.layers.TopHatOpening2D;

public class Config {

    private static final String SEP = ",";

    private static final String SETTINGS_FILE = "settings.properties";

    private static ClassLoader loader = Config.class.getClassLoader();

    private static Properties properties = null;

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

    private static Model buildModel() {
        return new Model(new TopHatOpening2D(), new Opening2D(Padding.VALID), new Flatten(),
                new Dense(Activation.TANH), new Dense(Activation.TANH), new Dense(Activation.SIGMOID));
    }

    public static synchronized Model loadModel() {
        Model model = buildModel();
        final String DIRECTORY = IJ.getDir("plugins").concat("/").concat(
                getProperties().getProperty("layers.subdirectory"));
        String[] layerIds = getProperties().getProperty("layers.ids").split(SEP);
        for (String layerId : layerIds) {
            int[] wShape = Arrays.stream(getProperties().getProperty(String.format("layers.layer_%s.shape", layerId))
                    .split(SEP)).mapToInt(s -> Integer.parseInt(s)).toArray();
            model.loadFrom(
                    HDF5Factory.openForReading(String.format("%s/layer_%s.h5", DIRECTORY, layerId)),
                    Integer.parseInt(layerId), wShape);
        }
        return model;
    }
}