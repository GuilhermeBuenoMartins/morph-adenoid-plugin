
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
                        FNArray x = getSample(stack.getProcessor(i + 1));
                        long startTime = System.currentTimeMillis();
                        float score = Config.getModel().predict(x).get(0, 0);
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
                FNArray x = getSample(ip);
                long startTime = System.currentTimeMillis();
                float score = Config.getModel().predict(x).get(0, 0);
                long time = System.currentTimeMillis() - startTime;
                IJ.getImage().setTitle(
                                String.format(TITLE_CLASSIFIED_IMAGE, BOUNDARY > score ? "NEGATIVE" : "POSITIVE", time,
                                                score));
        }

        private FNArray getSample(ImageProcessor ip) {
                float[] pixels = new float[INPUT_SHAPE[0] * INPUT_SHAPE[1] * INPUT_SHAPE[2]];
                float wScale = (float) INPUT_SHAPE[0] / ip.getWidth();
                float hScale = (float) INPUT_SHAPE[1] / ip.getHeight();
                int xi, yi;
                int p, offset;
                double rul, rur, rll, rlr;
                double gul, gur, gll, glr;
                double bul, bur, bll, blr;
                float xf, yf;
                for (int x = 0; x < INPUT_SHAPE[0]; x++) {
                        for (int y = 0; y < INPUT_SHAPE[1]; y++) {
                                xi = (int) (x / wScale);
                                yi = (int) (y / hScale);
                                xf = (x / wScale) - xi;
                                yf = (y / hScale) - yi;
                                p = ip.getPixel(xi, yi);
                                rul = (1 - xf) * (1 - yf) * getRed(p);
                                gul = (1 - xf) * (1 - yf) * getGreen(p);
                                bul = (1 - xf) * (1 - yf) * getBlue(p);
                                p = ip.getPixel(xi + 1, yi);
                                rur = xf * (1 - yf) * getRed(p);
                                gur = xf * (1 - yf) * getGreen(p);
                                bur = xf * (1 - yf) * getBlue(p);
                                p = ip.getPixel(xi, yi + 1);
                                rll = (1 - xf) * yf * getRed(p);
                                gll = (1 - xf) * yf * getGreen(p);
                                bll = (1 - xf) * yf * getBlue(p);
                                p = ip.getPixel(xi + 1, yi + 1);
                                rlr = xf * yf * getRed(p);
                                glr = xf * yf * getGreen(p);
                                blr = xf * yf * getBlue(p);
                                offset = INPUT_SHAPE[2] * (INPUT_SHAPE[0] * y + x);
                                pixels[offset] = (int) (rul + rur + rll + rlr + 0.5);
                                pixels[offset + 1] = (int) (gul + gur + gll + glr + 0.5);
                                pixels[offset + 2] = (int) (bul + bur + bll + blr + 0.5);
                        }
                }
                return FNArrayFactory.create(new int[] { 1, INPUT_SHAPE[1], INPUT_SHAPE[0], INPUT_SHAPE[2] }, pixels);
        }

        private int getRed(int pixel) {
                return (pixel & 0xff0000) >> 16;
        }

        private int getGreen(int pixel) {
                return (pixel & 0xff00) >> 8;
        }

        private int getBlue(int pixel) {
                return (pixel & 0xff);
        }
}
