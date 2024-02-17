package nn.utils;

public class Dilation {
    

    public synchronized static float operation(float[][][] x, float[][][][] w, int[] xIds, int wId) {
        float y = 0;
        for (int k = 0; k < w[0][0].length; k++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < w.length; i++) {
                for (int j = 0; j < w[0].length; j++) {
                    max = Math.max(x[xIds[0] + i][xIds[1] + j][k] + w[i][j][k][wId], max);
                }  
            }
            y += max;
        }
        return y;
    }

    public synchronized static float operation(float[][][] x, float[][][][] w, int[] xIds, int xCh, int wCh, int wId) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                max = Math.max(
                    x[xIds[0] + i][xIds[1] + j][xCh] 
                    + w[i][j][wCh][wId], max);
            }  
        }
        return max;
    }

}
