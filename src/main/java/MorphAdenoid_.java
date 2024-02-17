
import java.util.Arrays;
import java.util.stream.IntStream;

import ij.IJ;
import ij.ImagePlus;
import ij.gui.GenericDialog;
import ij.gui.MessageDialog;
import ij.plugin.PlugIn;
import ij.process.ImageProcessor;
import config.Config;
import nn.Model;
import nn.utils.Basics;

/**
 * Morph Adenoid!
 *
 */
public class MorphAdenoid_ implements PlugIn {


        private static Model model = null;

        private synchronized float[] convertPixels(float[] pixels, ImageProcessor ip) {
                int[] intPixels = (int[]) ip.getPixels();
                int chLength = ip.getHeight() * ip.getWidth();
                for (int i = 0; i < intPixels.length / 3; i += 3) {
                        pixels[i] = (float) intPixels[i];
                        pixels[chLength + i] = (float) intPixels[i + 1];
                        pixels[2 * chLength + i] = (float) intPixels[i + 2];
                }
                return pixels;
        }

        public synchronized float[][][][] getX(ImageProcessor ip) {
                int[] inputShape = Arrays.stream(Config.getProperties().getProperty("model.input_shape").split(","))
                                .mapToInt(s -> Integer.parseInt(s)).toArray();
                ImageProcessor xIp = ip.resize(inputShape[0], inputShape[1]);
                float[][][][] x = new float[1][inputShape[0]][inputShape[1]][3];
                return Basics.create(x, convertPixels(new float[inputShape[0] * inputShape[1] * 3], xIp));
        }

        public synchronized void classifyImage(Model model, ImagePlus imp) {
                long startTime = System.currentTimeMillis();
                float score = model.predict(getX(imp.getProcessor()))[0][0];
                long endTime = System.currentTimeMillis();
                boolean cls = score >= Float.parseFloat(Config.getProperties().getProperty("model.decision_boundary"));
                String title = String.format("Image classified in %d ms as \"%s\" with score = %f.",
                                endTime - startTime,
                                cls ? "Positive" : "Negative",
                                score);
                imp.setTitle(title);
        }

        public static int getStep(ImagePlus imp) {
                GenericDialog gd = new GenericDialog("Morph Adenoid");
                gd.addMessage("Please, set the step value for frame analyzing. In case of error, "
                                + "it will be consider the value " + Config.getProperties().getProperty("step.default") + ".");
                gd.addSlider("Step:", 1.0, imp.getStackSize(),
                                Double.parseDouble(Config.getProperties().getProperty("step.default")));
                gd.showDialog();
                return (int) gd.getNextNumber();
        }

        public synchronized void classifyStack(Model model, ImagePlus imp, int step) {
                IntStream.range(0, imp.getStackSize()).parallel().filter(i -> i % step == 0).forEach(i -> {
                        ImageProcessor ip = imp.getImageStack().getProcessor(i + 1);
                        long startTime = System.currentTimeMillis();
                        float score = model.predict(getX(ip))[0][0];
                        long endTime = System.currentTimeMillis();
                        boolean cls = score >= Float
                                        .parseFloat(Config.getProperties().getProperty("model.decision_boundary"));
                        if (cls) {
                                String title = String.format("Image classified in %d ms as \"Positive\" with score = %f.",
                                                endTime - startTime,
                                                score);
                                new ImagePlus(title, ip).show();
                        }
                });
        }

        @Override
        public void run(String arg) {
                if (model == null) { model = Config.loadModel(); }
                ImagePlus imp = IJ.getImage();
                boolean ok = IJ.showMessageWithCancel("Morph Adenoid", "Initializing plugin.");
                if (ok) {
                        if (imp.hasImageStack() && imp.isRGB()) {
                                classifyStack(model, imp, getStep(imp));
                                IJ.showMessage("The video was classified successful.");
                        } else if (imp.isRGB()) {
                                classifyImage(model, imp);
                                IJ.showMessage("The image was classified successful.");
                        } else {
                                IJ.error("Morph Adenoid",
                                                "What you want to processing must attend one of following specification:\n" +
                                                                "It must be a RGB image opened in ImageJ; or" +
                                                                "It must be a supported video opened in ImageJ.");
                        }
                }
        }
}
