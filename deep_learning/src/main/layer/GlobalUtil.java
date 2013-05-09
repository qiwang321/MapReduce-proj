package layer;
import java.lang.Math;
import java.util.Random;

public class GlobalUtil{
    public static int NUM_LAYER = 4;
    public static int NODES_INPUT = 784;
    public static double TRAIN_TIME = 80.0;    
    public static final int[] train_len = {5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949}; 
    public static final int[] test_len = {980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009};
    public static final int[] nodes_layer = {GlobalUtil.NODES_INPUT, 1000, 500, 250, 2};

    

    public static final float yita_w = (float) 0.05, yita_bv = (float) 0.05, yita_bh = (float) 0.05,
    yita_wt = (float) 5e-4, yita_bvt = (float) 5e-4, yita_bht = (float) 5e-4; // learning rates
    public static final float mu = (float) 0.5, reg = (float) 0.0002;

    public Random rd = new Random();
    /* function for computing sigmoid function
     * b: bias
     * w: weight vector
     * x: data vector
     */
    //compute the sigmoid of a layer
    public static void sigm(float[] res, float[] b, float[] W, float[] x, int n, int m, boolean dir) {
      if (dir == true) { //up
        for (int j = 0; j < n; j++) {
          res[j] = -b[j];
          for (int i = 0; i < m; i++)
            res[j] = res[j] - W[j*m + i] * x[i];
          res[j] = (float) (1.0 / (1 + Math.exp(res[j])));
        }
      }

      else { //down

        for (int i = 0; i < m; i++){
          res[i] = -b[i];
          for (int j = 0; j < n; j++)
            res[i] = res[i] - W[j*m + i] * x[j];
          res[i] = (float) (1.0 / (1 + Math.exp(res[i])));
        }
      }

    }


    //sample a Bernoulli r.v.
    int binrand(float p) {
      float t = rd.nextFloat();
      return t < p ? 1:0;
    }


    // compute the 2-norm distance between two vectors
    float dist(float[] x1, float[] x2, int len) {
      float sum = 0;
      for (int i = 0; i < len; i++)
        sum += (float)Math.pow(x1[i]-x2[i], 2);
      return (float) Math.sqrt(sum);
    }
    
};
