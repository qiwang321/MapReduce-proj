package bigsidemodel;
import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.io.Writable;



public class SuperModel implements Writable {
	
	// dimension parameters
	public int nModel;
	//private int nInput; // number of input nodes for the super layer
	public int nEach; // the number of outputs for each sub model
	public int NUM_LAYER = GlobalUtil.SUPER_NUM;
	public int NODES_INPUT;    
	//private final int[] train_len = GlobalUtil.train_len; 
	//private final int[] test_len = GlobalUtil.test_len;
	private int[] nodes_layer = GlobalUtil.super_layer;

	private final Random rd = new Random();  
	private float[][] sample_mem = new float[NUM_LAYER+1][]; //space storing the MCMC samples
	private float[][] weights = new float[NUM_LAYER+1][]; //space storing the updating weights (first is not used)
	private float[][] bh = new float[NUM_LAYER+1][]; // hidden layer biases (rbm)
	private float[][] bv = new float[NUM_LAYER+1][]; // visible layer biases (rbm)  

  private float[][] inc_w = new float[NUM_LAYER+1][];//[nodes_layer[layer_ind-1]*nodes_layer[layer_ind]]; // previous increase of weights
  private float[][] inc_bv = new float[NUM_LAYER+1][];//new float[nodes_layer[layer_ind-1]];
  private float[][] inc_bh = new float[NUM_LAYER+1][];// float[nodes_layer[layer_ind]];
  
	// training parameters
	private float yita_w = GlobalUtil.yita_w, yita_bv = GlobalUtil.yita_bv, yita_bh = GlobalUtil.yita_bh,
			yita_wt = GlobalUtil.yita_wt, yita_bvt = GlobalUtil.yita_bvt, yita_bht = GlobalUtil.yita_bht; // learning rates
	private float mu = GlobalUtil.mu, reg = GlobalUtil.reg;

	public SuperModel(int n) {
		nModel = n;
		nEach = GlobalUtil.nodes_layer[GlobalUtil.NUM_LAYER];
		NODES_INPUT = nModel * nEach;
		nodes_layer = GlobalUtil.super_layer;
		nodes_layer[0] = NODES_INPUT;
		sample_mem[0] = new float[NODES_INPUT];
		for (int k = 1; k <= NUM_LAYER; k++) {
			weights[k] = new float[nodes_layer[k] * nodes_layer[k-1]];
			inc_w[k] = new float[nodes_layer[k] * nodes_layer[k-1]];
			for (int i = 0; i < nodes_layer[k] * nodes_layer[k-1]; i++)
				weights[k][i] = 0.1f * (float)rd.nextGaussian();
			sample_mem[k] = new float[nodes_layer[k]];
			bh[k] = new float[nodes_layer[k]];
			inc_bh[k] = new float[nodes_layer[k]];
			for (int i = 0; i < nodes_layer[k]; i++)
				bh[k][i] = 0.0f;
			bv[k] = new float[nodes_layer[k-1]];
	    inc_bv[k] = new float[nodes_layer[k-1]];

			for (int i = 0; i < nodes_layer[k-1]; i++)
				bv[k][i] = 0.0f;
		}
		
    for (int k = 1; k <= NUM_LAYER; k++) {
      Arrays.fill(inc_w[k],0);
      Arrays.fill(inc_bv[k], 0);
      Arrays.fill(inc_bh[k], 0);
    }
  }

	public void train(float[] data) throws IOException {
	    for (int i=0;i<nModel*nEach;i++)
	        sample_mem[0][i] = data[i];
			
			// acquired data for super model, standard training
			for (int layer_ind = 1; layer_ind <= NUM_LAYER; layer_ind++)
				work_update(layer_ind);
	}


	@Override
	public void readFields(DataInput in) throws IOException {
		// TODO Auto-generated method stub
    nModel = in.readInt();
    nEach = in.readInt();
    NODES_INPUT = in.readInt();
   
    nodes_layer = GlobalUtil.super_layer;
    nodes_layer[0] = NODES_INPUT;
    
    sample_mem[0] = new float[NODES_INPUT];
    for (int k = 1; k <= NUM_LAYER; k++) {
      weights[k] = new float[nodes_layer[k] * nodes_layer[k-1]];
      inc_w[k] = new float[nodes_layer[k] * nodes_layer[k-1]];

      sample_mem[k] = new float[nodes_layer[k]];
      bh[k] = new float[nodes_layer[k]];
      inc_bh[k] = new float[nodes_layer[k]];
      
      bv[k] = new float[nodes_layer[k-1]];
      inc_bv[k] = new float[nodes_layer[k-1]];
    }

    for (int k = 1; k <= NUM_LAYER; k++) {
      for (int i = 0; i < nodes_layer[k] * nodes_layer[k-1]; i++)
        weights[k][i]=in.readFloat();
    }
    
    for (int k = 1; k <= NUM_LAYER; k++) {
      for (int i = 0; i< nodes_layer[k]; i++) 
          bh[k][i] = in.readFloat();
    }
    
    for (int k = 1; k <= NUM_LAYER; k++) {
      for (int i = 0; i< nodes_layer[k-1]; i++) 
          bv[k][i] = in.readFloat();
    }
	}

	@Override
	public void write(DataOutput out) throws IOException {
		// TODO Auto-generated method stub
	  out.writeInt(nModel);
	  out.writeInt(nEach);
	  out.writeInt(NODES_INPUT);
	 
    nodes_layer = GlobalUtil.super_layer;
    nodes_layer[0] = NODES_INPUT;
    
    for (int k = 1; k <= NUM_LAYER; k++) {
      for (int i = 0; i < nodes_layer[k] * nodes_layer[k-1]; i++)
        out.writeFloat(weights[k][i]);
    }
    
    for (int k = 1; k <= NUM_LAYER; k++) {
      for (int i = 0; i< nodes_layer[k]; i++) 
          out.writeFloat(bh[k][i]);
    }
    
    for (int k = 1; k <= NUM_LAYER; k++) {
      for (int i = 0; i< nodes_layer[k-1]; i++) 
          out.writeFloat(bv[k][i]);
    }
    
	}
	
	@Override 
	
	public String toString(){return "finished";}
	
	void work_update(int layer_ind){
		float[] x0 = new float[nodes_layer[layer_ind - 1]]; // data
		float[] h0 = new float[nodes_layer[layer_ind]];  // hidden
		float[] x1 = new float[nodes_layer[layer_ind - 1]];
		float[] h1 = new float[nodes_layer[layer_ind]];


		for (int i = 0; i < nodes_layer[layer_ind - 1]; i++)
			x0[i] = sample_mem[layer_ind - 1][i];

		if (layer_ind != NUM_LAYER) { // normal layer        

			//perform real computation
			GlobalUtil.sigm(h0, bh[layer_ind], weights[layer_ind], x0,
					nodes_layer[layer_ind], nodes_layer[layer_ind-1], true);// up sampling

			for (int j = 0; j < nodes_layer[layer_ind]; j++)
				sample_mem[layer_ind][j] = h0[j];

			for (int i = 0; i < nodes_layer[layer_ind]; i++) {
				if (rd.nextFloat() < h0[i])
					h0[i] = 1;
				else
					h0[i] = 0;
			}


			GlobalUtil.sigm(x1, bv[layer_ind], weights[layer_ind], h0,
					nodes_layer[layer_ind], nodes_layer[layer_ind-1], false);// down sampling

					GlobalUtil.sigm(h1, bh[layer_ind], weights[layer_ind], x1,
							nodes_layer[layer_ind], nodes_layer[layer_ind-1], true);

					for (int j = 0; j < nodes_layer[layer_ind]; j++)
						for (int i = 0; i < nodes_layer[layer_ind-1]; i++) {
							inc_w[layer_ind][j*nodes_layer[layer_ind-1] + i] = mu * inc_w[layer_ind][j*nodes_layer[layer_ind-1] + i]
									+ yita_w * (h0[j]*x0[i] - h1[j]*x1[i] - reg * weights[layer_ind][j*nodes_layer[layer_ind-1] + i]);
							weights[layer_ind][j*nodes_layer[layer_ind-1] + i] =
									weights[layer_ind][j*nodes_layer[layer_ind-1] + i]
											+inc_w[layer_ind][j*nodes_layer[layer_ind-1] + i];
						}

					for (int j = 0; j < nodes_layer[layer_ind]; j++) {
						inc_bh[layer_ind][j] = mu * inc_bh[layer_ind][j] + yita_bh*(h0[j] - h1[j] - reg * bh[layer_ind][j]);
						bh[layer_ind][j] = bh[layer_ind][j] + inc_bh[layer_ind][j];
					}

					for (int i = 0; i < nodes_layer[layer_ind-1]; i++) {
						inc_bv[layer_ind][i] = mu * inc_bv[layer_ind][i] + yita_bv*(x0[i] - x1[i] - reg * bv[layer_ind][i]);
						bv[layer_ind][i] = bv[layer_ind][i] + inc_bv[layer_ind][i];
					}
					// print the layer input data (just for testing)
		}
		else { // top layer
			//perform real computation
			for (int j = 0; j < nodes_layer[NUM_LAYER]; j++) {
				h0[j] = bh[NUM_LAYER][j];
				for (int i = 0; i < nodes_layer[NUM_LAYER-1]; i++)
					h0[j] = h0[j] + weights[NUM_LAYER][j*nodes_layer[NUM_LAYER-1] + i] * x0[i];
			}

			for (int j = 0; j < nodes_layer[layer_ind]; j++)
				sample_mem[layer_ind][j] = h0[j];


			GlobalUtil.sigm(x1, bv[layer_ind], weights[NUM_LAYER], h0,
					nodes_layer[layer_ind], nodes_layer[layer_ind-1], false);// down sampling

			for (int j = 0; j < nodes_layer[NUM_LAYER]; j++) {
				h1[j] = bh[NUM_LAYER][j];
				for (int i = 0; i < nodes_layer[NUM_LAYER-1]; i++)
					h1[j] = h1[j] + weights[NUM_LAYER][j*nodes_layer[NUM_LAYER-1] + i] * x1[i];
			}

			for (int j = 0; j < nodes_layer[layer_ind]; j++)
				for (int i = 0; i < nodes_layer[layer_ind-1]; i++) {
					inc_w[layer_ind][j*nodes_layer[layer_ind-1] + i] = mu * inc_w[layer_ind][j*nodes_layer[layer_ind-1] + i]
							+ yita_wt * (h0[j]*x0[i] - h1[j]*x1[i] - reg * weights[layer_ind][j*nodes_layer[layer_ind-1] + i]);
					weights[layer_ind][j*nodes_layer[layer_ind-1] + i] =
							weights[layer_ind][j*nodes_layer[layer_ind-1] + i]
									+inc_w[layer_ind][j*nodes_layer[layer_ind-1] + i];
				}

			for (int j = 0; j < nodes_layer[layer_ind]; j++) {
				inc_bh[layer_ind][j] = mu * inc_bh[layer_ind][j] + yita_bht*(h0[j] - h1[j] - reg * bh[layer_ind][j]);
				bh[layer_ind][j] = bh[layer_ind][j] + inc_bh[layer_ind][j];
			}

			for (int i = 0; i < nodes_layer[layer_ind-1]; i++) {
				inc_bv[layer_ind][i] = mu * inc_bv[layer_ind][i] + yita_bvt*(x0[i] - x1[i] - reg * bv[layer_ind][i]);
				bv[layer_ind][i] = bv[layer_ind][i] + inc_bv[layer_ind][i];
			}
			// print the layer input data (just for testing)
		}
	}
	
	
	
  public float[] test(float[] test_records){
    for (int i=0;i<NODES_INPUT;i++) 
        sample_mem[0][i]=test_records[i];
    

    for (int k = 1; k < NUM_LAYER; k++)
      GlobalUtil.sigm(sample_mem[k], bh[k], weights[k], sample_mem[k-1],
          nodes_layer[k], nodes_layer[k-1], true);
    
    for (int j = 0; j < nodes_layer[NUM_LAYER]; j++) {
      sample_mem[NUM_LAYER][j] = -bh[NUM_LAYER][j];
      for (int i = 0; i < nodes_layer[NUM_LAYER-1]; i++)
        sample_mem[NUM_LAYER][j] = sample_mem[NUM_LAYER][j]
                                   - weights[NUM_LAYER][j*nodes_layer[NUM_LAYER-1] + i] * sample_mem[NUM_LAYER-1][i];
    }
    float[] result = new float[nodes_layer[NUM_LAYER]];
    for (int j = 0; j < nodes_layer[NUM_LAYER]; j++)
      result[j] = sample_mem[NUM_LAYER][j];
    return result;
  }
}
