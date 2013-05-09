package biglayer;
/*
 * Cloud9: A Hadoop toolkit for working with big data
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you
 * may not use this file except in compliance with the License. You may
 * obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */
// package model;


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

  
  private int ID;
//	private ArrayListOfFloatsWritable[] weights = new ArrayListOfFloatsWritable[GlobalUtil.NUM_LAYER+1];
//  private ArrayListOfFloatsWritable[] bv = new ArrayListOfFloatsWritable[GlobalUtil.NUM_LAYER+1];
//	private ArrayListOfFloatsWritable[] bh = new ArrayListOfFloatsWritable[GlobalUtil.NUM_LAYER+1];
  private int layer_ind;
  private float[] data;
  private float[][] weights = new float[GlobalUtil.NUM_LAYER+1][]; //space storing the updating weights (first is not used)
  private float[][] bh = new float[GlobalUtil.NUM_LAYER+1][]; // hidden layer biases (rbm)
  private float[][] bv = new float[GlobalUtil.NUM_LAYER+1][]; // visible layer biases (rbm)

	public ModelNode() {
	}

	public int getID(){
	  return ID;
	}
	public int getLayerInd(){
	   return layer_ind;
	}
	public float[] getData(){
	  return data;
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

  public void setID(int id){
    this.ID = id;
  }
  public void setLayerInd(int layer_id) {
    this.layer_ind = layer_id;
  }
  public void setData(float[] data) {
    this.data = data;
  }
	public void setWeight(float[][] weight) {
		this.weights = weight;
	}
  public void setBH(float[][] bh) {
    this.bh = bh;
  }
  public void setBV(float[][] bv) {
    this.bv = bv;
  }

	

	/**
	 * Deserializes this object.
	 *
	 * @param in source for raw byte representation
	 */
	@Override
	public void readFields(DataInput in) throws IOException {
	  ID = in.readInt();
    layer_ind = in.readInt();
    if (ID>=0) {
        data = new float[GlobalUtil.nodes_layer[layer_ind]];
        for (int k=0;k<GlobalUtil.nodes_layer[layer_ind];k++) 
           data[k] = in.readFloat();
        return;
    }
    
	  for (int k = 1; k <= GlobalUtil.NUM_LAYER; k++) {
      weights[k] = new float[GlobalUtil.nodes_layer[k] * GlobalUtil.nodes_layer[k-1]];
      bh[k] = new float[GlobalUtil.nodes_layer[k]];
      bv[k] = new float[GlobalUtil.nodes_layer[k-1]];
    }

	  for (int k=1; k<=GlobalUtil.NUM_LAYER;k++) {
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
	}

	/**
	 * Serializes this object.
	 *
	 * @param out where to write the raw byte representation
	 */
	@Override
	public void write(DataOutput out) throws IOException {
	  out.writeInt(ID);
	  out.writeInt(layer_ind);
	  if (ID>=0) {
	      for (int k=0;k<GlobalUtil.nodes_layer[layer_ind];k++) 
	        out.writeFloat(data[k]);
	      return;
	  }
	  
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
  	float[] res = new float[GlobalUtil.nodes_layer[1]];
  	float[] res_prev;
  	res_prev = data;
  	int n, m;
  	for (int i = 1; i <= GlobalUtil.NUM_LAYER; i++) {
  		res = new float[GlobalUtil.nodes_layer[i]];
  		n = GlobalUtil.nodes_layer[i];
  		m = GlobalUtil.nodes_layer[i-1];
  		GlobalUtil.sigm(res, bh[i], weights[i], res_prev, n, m, true);
  		res_prev = res;
  	}
  	return res;
  }
  
  public float[] test(float[] test_records){
    float[] res;
    float[] res_prev;
    res_prev = test_records;
    
    for (int k = 1; k < GlobalUtil.NUM_LAYER; k++) {
      res = new float[GlobalUtil.nodes_layer[k]];
      GlobalUtil.sigm(res, bh[k], weights[k], res_prev,
          GlobalUtil.nodes_layer[k], GlobalUtil.nodes_layer[k-1], true);
      res_prev = res;
    }
    res = new float[GlobalUtil.nodes_layer[GlobalUtil.NUM_LAYER]];
    
    for (int j = 0; j < GlobalUtil.nodes_layer[GlobalUtil.NUM_LAYER]; j++) {
      res[j] = -bh[GlobalUtil.NUM_LAYER][j];
      for (int i = 0; i < GlobalUtil.nodes_layer[GlobalUtil.NUM_LAYER-1]; i++)
        res[j] = res[j] - weights[GlobalUtil.NUM_LAYER][j*GlobalUtil.nodes_layer[GlobalUtil.NUM_LAYER-1] + i] * res_prev[i];
    }
    return res;
  }
  
}
