package nn.utils;

public class Erosion {

    public synchronized static float operation(float[][][] x, float[][][][] w, int[] xIds, int wId) {
        float y = 0;
        for (int k = 0; k < w[0][0].length; k++) {
            float min = Float.POSITIVE_INFINITY;
            for (int i = 0; i < w.length; i++) {
                for (int j = 0; j < w[0].length; j++) {
                    min = Math.min(x[xIds[0] + i][xIds[1] + j][k] - w[w.length - 1 - i][w[0].length - 1 - j][k][wId], min);
                }  
            }
            y += min;
        }
        return y;
    }

    public synchronized static float operation(float[][][] x, float[][][][] w, int[] xIds, int xCh, int wId) {
        float min = Float.POSITIVE_INFINITY;
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                min = Math.min(
                    x[xIds[0] + i][xIds[1] + j][xCh] 
                    - w[w.length - 1 - i][w[0].length - 1 - j][xCh][wId], min);
            }  
        }
        return min;
    }
}
