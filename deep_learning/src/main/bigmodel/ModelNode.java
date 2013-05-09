package bigmodel;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutput;
import java.io.DataOutputStream;
import java.io.IOException;
import org.apache.hadoop.io.Writable;

import edu.umd.cloud9.io.array.ArrayListOfFloatsWritable;
import edu.umd.cloud9.io.array.ArrayListOfIntsWritable;

/**
 * Representation of a graph node for PageRank. 
 *
 * @author Jimmy Lin
 * @author Michael Schatz
 */
public class ModelNode implements Writable {

  private float[][] sample_mem = new float[GlobalUtil.NUM_LAYER+1][]; //space storing the MCMC samples
	private float[][] weights = new float[GlobalUtil.NUM_LAYER+1][];
  private float[][] bv = new float[GlobalUtil.NUM_LAYER+1][];
	private float[][] bh = new float[GlobalUtil.NUM_LAYER+1][];
	
	private float[] weights_field;
	private float[] bh_field;
	private float[] bv_field;
	
	public ModelNode() {
	}

	public float[][] getWeight() {
		return weights;
	}

  public float[][] getBH() {
    return bh;
  }

  public float[][] getBV() {
    return bv;
  }

	public void setWeight(float[][] weight, float[] weights_field) {
		this.weights = weight;
		this.weights_field = weights_field;
	}
  public void setBH(float[][] bh, float[] bh_field) {
    this.bh = bh;
    this.bh_field = bh_field;
  }
  public void setBV(float[][] bv, float[] bv_field) {
    this.bv = bv;
    this.bv_field = bv_field;
  }

	

	/**
	 * Deserializes this object.
	 *
	 * @param in source for raw byte representation
	 */
	@Override
	public void readFields(DataInput in) throws IOException {
    sample_mem[0] = new float[GlobalUtil.NODES_INPUT];
    for (int k = 1; k <= GlobalUtil.NUM_LAYER; k++) {
      weights[k] = new float[GlobalUtil.nodes_layer[k] * GlobalUtil.nodes_layer[k-1]];
      sample_mem[k] = new float[GlobalUtil.nodes_layer[k]];
      bh[k] = new float[GlobalUtil.nodes_layer[k]];
      bv[k] = new float[GlobalUtil.nodes_layer[k-1]];
    }

    for (int k=1; k<= GlobalUtil.NUM_LAYER;k++) {
      for (int i = 0; i < GlobalUtil.nodes_layer[k-1] * GlobalUtil.nodes_layer[k]; i++)
          weights[k][i]=in.readFloat();
    }
  
    for (int k = 1; k <= GlobalUtil.NUM_LAYER; k++) {
      for (int i = 0; i< GlobalUtil.nodes_layer[k]; i++) 
          bh[k][i] = in.readFloat();
    }
    
    for (int k = 1; k <= GlobalUtil.NUM_LAYER; k++) {
      for (int i = 0; i< GlobalUtil.nodes_layer[k-1]; i++) 
          bv[k][i] = in.readFloat();
    }
    

    weights_field = new float[GlobalUtil.NODES_INPUT * GlobalUtil.window * GlobalUtil.window];
    bh_field = new float[GlobalUtil.NODES_INPUT];
    bv_field = new float[GlobalUtil.sizeInput];
    
    for (int j = 0; j < GlobalUtil.NODES_INPUT * GlobalUtil.window * GlobalUtil.window; j++)
      weights_field[j] = in.readFloat();
    for (int j = 0; j < GlobalUtil.NODES_INPUT; j++)
      bh_field[j] = in.readFloat();
    for (int j = 0; j < GlobalUtil.sizeInput; j++)
      bv_field[j] = in.readFloat();    
	}

	/**
	 * Serializes this object.
	 *
	 * @param out where to write the raw byte representation
	 */
	@Override
	public void write(DataOutput out) throws IOException {
    
    for (int k = 1; k <= GlobalUtil.NUM_LAYER; k++) {
      for (int i = 0; i < GlobalUtil.nodes_layer[k] * GlobalUtil.nodes_layer[k-1]; i++)
        out.writeFloat(weights[k][i]);
    }
    
    for (int k = 1; k <= GlobalUtil.NUM_LAYER; k++) {
      for (int i = 0; i< GlobalUtil.nodes_layer[k]; i++) 
          out.writeFloat(bh[k][i]);
    }
    
    for (int k = 1; k <= GlobalUtil.NUM_LAYER; k++) {
      for (int i = 0; i< GlobalUtil.nodes_layer[k-1]; i++) 
          out.writeFloat(bv[k][i]);
    }
 
    for (int j = 0; j < GlobalUtil.NODES_INPUT * GlobalUtil.window * GlobalUtil.window; j++)
      out.writeFloat(weights_field[j]);
    for (int j = 0; j < GlobalUtil.NODES_INPUT; j++)
      out.writeFloat(bh_field[j]);
    for (int j = 0; j < GlobalUtil.sizeInput; j++)
      out.writeFloat(bv_field[j]);    
	}

	@Override
	public String toString() {
		String output = "";
		for (int k = 1; k <= GlobalUtil.NUM_LAYER; k++) {
			output = output + "weights[" + k + "]:\n";
			for (int j = 0; j < GlobalUtil.nodes_layer[k]; j++) {
				for (int i = 0; i < GlobalUtil.nodes_layer[k-1]; i++) {
					output = output + weights[k][GlobalUtil.nodes_layer[k-1]*j + i] + " ";
				}
				output = output + "\n";
			}
		}
		for (int k = 1; k <= GlobalUtil.NUM_LAYER; k++) {
			output = output + "bias[" + k + "]:\n";
			for (int j = 0; j < GlobalUtil.nodes_layer[k]; j++) {
				output = output + bh[k][j] + " ";
			}
			output = output + "\n";
		}
		return output;
	}


  /**
   * Returns the serialized representation of this object as a byte array.
   *
   * @return byte array representing the serialized representation of this object
   * @throws IOException
   */
  public byte[] serialize() throws IOException {
    ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
    DataOutputStream dataOut = new DataOutputStream(bytesOut);
    write(dataOut);

    return bytesOut.toByteArray();
  }

  /**
   * Creates object from a <code>DataInput</code>.
   *
   * @param in source for reading the serialized representation
   * @return newly-created object
   * @throws IOException
   */
  public static ModelNode create(DataInput in) throws IOException {
    ModelNode m = new ModelNode();
    m.readFields(in);

    return m;
  }

  /**
   * Creates object from a byte array.
   *
   * @param bytes raw serialized representation
   * @return newly-created object
   * @throws IOException
   */
  public static ModelNode create(byte[] bytes) throws IOException {
    return create(new DataInputStream(new ByteArrayInputStream(bytes)));
  }
  
  
  
  public float[] sim(float[] data) {
  	
  	float[] res = new float[GlobalUtil.NODES_INPUT];
  	float[] res_prev;
  	int n, m;
  	
  	int numWindows = GlobalUtil.numWindows;
  	int window = GlobalUtil.window;
  	int step = GlobalUtil.step;
  	int imageSize = GlobalUtil.imageSize;
  	int raw2d = GlobalUtil.raw2d;
  	int new2d = GlobalUtil.new2d;
  	
/*  	for (int l = 0; l < 3; l++)
  	for (int i = 0; i < numWindows; i++)
			for (int j = 0; j < numWindows; j++) {
			int ind_h = l*new2d + i*numWindows + j;
			res[ind_h] = -bh_field[ind_h];
			for (int s = 0; s < window; s++)
				for (int t = 0; t < window; t++) {
					int ind_v = l*raw2d + (i*step+s)*imageSize + j*step+t;
					int ind_w = ind_h*window*window + s*window+t;
					res[ind_h] = res[ind_h] - weights_field[ind_w] * data[ind_v];
				}
			res[ind_h] = 1.0f / (1.0f + (float)Math.exp(res[ind_h]));
		}*/
  	
		for (int l = 0; l < 3; l++) 
		for (int i = 0; i < numWindows; i++)
			for (int j = 0; j < numWindows; j++) {
			int ind_h = l*new2d + i*numWindows + j;
			res[ind_h] = 0;
			for (int s = 0; s < window; s++)
				for (int t = 0; t < window; t++) {
					int ind_v = l*raw2d + (i*step+s)*imageSize + j*step+t;
					int ind_w = ind_h*window*window + s*window+t;
					res[ind_h] = res[ind_h] + data[ind_v];
				}
			res[ind_h] /= window*window;
		}
  	
  	res_prev = res;
  	
  	for (int i = 1; i <= GlobalUtil.NUM_LAYER; i++) {
  		res = new float[GlobalUtil.nodes_layer[i]];
  		n = GlobalUtil.nodes_layer[i];
  		m = GlobalUtil.nodes_layer[i-1];
  		GlobalUtil.sigm(res, bh[i], weights[i], res_prev, n, m, true);
  		res_prev = res;
  	}
  	return res;
  }
}
