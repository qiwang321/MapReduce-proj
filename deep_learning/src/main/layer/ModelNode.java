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
package layer;


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

	private ArrayListOfFloatsWritable[] weights = new ArrayListOfFloatsWritable[GlobalUtil.NUM_LAYER+1];
  private ArrayListOfFloatsWritable[] bv = new ArrayListOfFloatsWritable[GlobalUtil.NUM_LAYER+1];
	private ArrayListOfFloatsWritable[] bh = new ArrayListOfFloatsWritable[GlobalUtil.NUM_LAYER+1];;
	
	public ModelNode() {
	}

	public ArrayListOfFloatsWritable[] getWeight() {
		return weights;
	}

  public ArrayListOfFloatsWritable[] getBH() {
    return bh;
  }

  public ArrayListOfFloatsWritable[] getBV() {
    return bv;
  }

	public void setWeight(ArrayListOfFloatsWritable[] weight) {
		this.weights = weight;
	}
  public void setBH(ArrayListOfFloatsWritable[] bh) {
    this.bh = bh;
  }
  public void setBV(ArrayListOfFloatsWritable[] bv) {
    this.bv = bv;
  }

	

	/**
	 * Deserializes this object.
	 *
	 * @param in source for raw byte representation
	 */
	@Override
	public void readFields(DataInput in) throws IOException {
	  for (int i=1; i<GlobalUtil.NUM_LAYER+1;i++) {
	    weights[i] = new ArrayListOfFloatsWritable();
	    weights[i].readFields(in);
	  }

    for (int i=1; i<GlobalUtil.NUM_LAYER+1;i++) {
      bv[i] = new ArrayListOfFloatsWritable();
      bv[i].readFields(in);
    }
    
    for (int i=1; i<GlobalUtil.NUM_LAYER+1;i++) {
      bh[i] = new ArrayListOfFloatsWritable();
      bh[i].readFields(in);    
    }
	}

	/**
	 * Serializes this object.
	 *
	 * @param out where to write the raw byte representation
	 */
	@Override
	public void write(DataOutput out) throws IOException {
    for (int i=1; i<GlobalUtil.NUM_LAYER+1;i++) 
      weights[i].write(out);
		
    for (int i=1; i<GlobalUtil.NUM_LAYER+1;i++) 
      bv[i].write(out);
		  
    for (int i=1; i<GlobalUtil.NUM_LAYER+1;i++) 
      bh[i].write(out);
	}

	@Override
	public String toString() {
		return String.format("{%s %s %s}",
				weights.toString(), bv.toString(), bh.toString());
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
}
