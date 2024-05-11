
import java.util.Arrays;
import java.util.stream.IntStream;

import config.Config;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.GenericDialog;
import ij.plugin.PlugIn;
import ij.process.ImageProcessor;
import neural.network.core.FNArray;
import neural.network.core.FNArrayFactory;

/**
 * Morph Adenoid!
 *
 */
public class MorphAdenoid_ implements PlugIn {

        private static final String PLUGIN_NAME = "Morph Adenoid";

        private static final String MSG_ERROR = "Not was possible processing. Verify whether what you want to process have RGB layers.";

        private static final String MSG_CONTINUE = "The plugin will be initialize. Are you sure to wish continue?";

        private static final String MSG_STEP = "Please, set the step value by frame analysis. In case any error, it will be considered the value %.0f";

        private static final String TITLE_CLASSIFIED_IMAGE = "This image was classified as %s in %d ms with score = %f.";

        private static final String MSG_CLASSIFICATION = "The classification was done successful!";

        private static final double MIN_STEP_VALUE = 1;

        private static final double DEFAULT_STEP_VALUE = Double
                        .parseDouble(Config.getProperties().getProperty("step.default"));

        private static final float BOUNDARY = Float
                        .parseFloat(Config.getProperties().getProperty("model.decision_boundary"));

        private static final int[] INPUT_SHAPE = Arrays
                        .stream(Config.getProperties().getProperty("model.input_shape").split(Config.SEP))
                        .mapToInt(s -> Integer.parseInt(s)).toArray();

        @Override
        public void run(String arg) {
                if (!IJ.getImage().isRGB()) {
                        IJ.error(PLUGIN_NAME, MSG_ERROR);
                } else {
                        if (IJ.showMessageWithCancel(PLUGIN_NAME, MSG_CONTINUE)) {
                                if (!IJ.getImage().hasImageStack()) {
                                        classify(IJ.getImage().getProcessor());
                                } else {
                                        classify(IJ.getImage().getImageStack());
                                }
                                IJ.showMessage(PLUGIN_NAME, MSG_CLASSIFICATION);
                        }
                }
        }

        private void classify(ImageStack stack) {
                final int step = getStep(stack);
                IntStream.range(0, stack.getSize()).filter(i -> i % step == 0).forEach(i -> {
                        long startTime = System.currentTimeMillis();
                        float score = Config.getModel().predict(getSample(stack.getProcessor(i + 1))).get(0, 0);
                        long time = System.currentTimeMillis() - startTime;
                        if (score >= BOUNDARY) {
                                new ImagePlus(String.format(TITLE_CLASSIFIED_IMAGE, "POSITIVE", time, score),
                                                stack.getProcessor(i + 1)).show();
                        }
                        IJ.showProgress(i + 1, stack.getSize());
                });
                IJ.showProgress(1);
        }

        private int getStep(ImageStack stack) {
                GenericDialog gd = new GenericDialog(PLUGIN_NAME);
                gd.addMessage(String.format(MSG_STEP, DEFAULT_STEP_VALUE));
                gd.addSlider("Step value:", MIN_STEP_VALUE, stack.getSize(), DEFAULT_STEP_VALUE);
                gd.showDialog();
                return (int) gd.getNextNumber();
        }

        private void classify(ImageProcessor ip) {
                long startTime = System.currentTimeMillis();
                float score = Config.getModel().predict(getSample(ip)).get(0, 0);
                long time = System.currentTimeMillis() - startTime;
                IJ.getImage().setTitle(
                                String.format(TITLE_CLASSIFIED_IMAGE, BOUNDARY > score ? "NEGATIVE" : "POSITIVE", time,
                                                score));
        }

        private FNArray getSample(ImageProcessor ip) {
                int chLength = INPUT_SHAPE[0] * INPUT_SHAPE[1];
                float[] values = new float[chLength * INPUT_SHAPE[2]];
                int[] pixels = (int[]) ip.resize(INPUT_SHAPE[0], INPUT_SHAPE[1]).getPixels();
                for (int i = 0; i < pixels.length / INPUT_SHAPE[2]; i += INPUT_SHAPE[2]) {
                        values[i] = (float) pixels[i];
                        values[chLength + i] = (float) pixels[i + 1];
                        values[2 * chLength + i] = (float) pixels[i + 2];
                }
                return FNArrayFactory.create(
                                new int[] { 1, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2] },
                                values);
        }

}
