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

//package edu.umd.cloud9.example.bigram;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.channels.GatheringByteChannel;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;
import java.util.StringTokenizer;
import java.util.Vector;
import java.util.Arrays;


import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import org.mortbay.log.Log;

import edu.umd.cloud9.io.array.ArrayListOfFloatsWritable;
import edu.umd.cloud9.io.pair.PairOfInts;
import edu.umd.cloud9.io.pair.PairOfStrings;

public class AutoCoder extends Configured implements Tool {
	private static final Logger LOG = Logger.getLogger(AutoCoder.class);

	
	 protected static class MyMapper0 extends Mapper<Text, Text, PairOfInts, ModelNode> {
	    private static final PairOfInts modelID_Type = new PairOfInts();
	    private static final ModelNode modelStructure = new ModelNode();   
	    private static final ModelNode modelData = new ModelNode();

	    private static final Random rd = new Random();  

	    private static int layer_ind=0;
	    private static int numReducers = 0;

	    public void setup(Context context) throws IOException{
	      layer_ind = 0;
	      numReducers = context.getConfiguration().getInt("num_reduce_task", 1);
	    }

	    @Override
	    public void map(Text key, Text value, Context context)
	        throws IOException, InterruptedException {
	      String line = value.toString();
	      if (line.length() == 0) return;
	      StringTokenizer itr = new StringTokenizer(line);
	      float[] data=new float[GlobalUtil.NODES_INPUT];

	      int tot=0;
	      while (itr.hasMoreTokens()){
	        String curr = itr.nextToken();
	        data[tot] = Float.parseFloat(curr) / 255.0f;
	        tot++;
	      }
	      int ID = Integer.parseInt(key.toString());
	      modelData.setID(ID);
	      modelData.setLayerInd(layer_ind);
	      modelData.setData(data);
	      
	      if (ID == 0) {
	        modelID_Type.set(rd.nextInt(numReducers) % numReducers, ID); //rd.nextInt()%numReducers, 1);
	        context.write(modelID_Type, modelData);
	      }
	      else {
	        for (int i=0;i<numReducers;i++){
	          modelID_Type.set(i, ID); //rd.nextInt()%numReducers, 1);
	          context.write(modelID_Type, modelData);    
	        }   
	      }
	    }
	  }



	  protected static class MyReducer0 extends Reducer<PairOfInts, ModelNode, PairOfInts, ModelNode> {
	    private static final Random rd = new Random();  
	    private static float[][] weights = new float[GlobalUtil.NUM_LAYER+1][]; //space storing the updating weights (first is not used)
	    private static float[][] bh = new float[GlobalUtil.NUM_LAYER+1][]; // hidden layer biases (rbm)
	    private static float[][] bv = new float[GlobalUtil.NUM_LAYER+1][]; // visible layer biases (rbm)  
	    
	    private static final PairOfInts modelID_Type = new PairOfInts();
	    private static final ModelNode modelStructure = new ModelNode();   
	    private static final ModelNode modelData = new ModelNode();
	    private static int reducer_id = 0;
      private static int layer_ind=0;
      private static int count = 0;
	    
	    public void setup(Context context) throws IOException{
	        for (int k = 1; k < GlobalUtil.NUM_LAYER + 1; k++) {
	          weights[k] = new float[GlobalUtil.nodes_layer[k-1] * GlobalUtil.nodes_layer[k]];
	          bv[k] = new float[GlobalUtil.nodes_layer[k-1]];
	          bh[k] = new float[GlobalUtil.nodes_layer[k]];
	        }

	        for (int k = 1; k < GlobalUtil.NUM_LAYER + 1; k++) 
	          for (int j = 0; j < GlobalUtil.nodes_layer[k-1] * GlobalUtil.nodes_layer[k]; j++)
	            weights[k][j] = 0.1f * (float)rd.nextGaussian();

	        for (int k = 1; k < GlobalUtil.NUM_LAYER + 1; k++) 
	          for (int j = 0; j < GlobalUtil.nodes_layer[k-1]; j++)
	            bv[k][j] = 0.0f;

	        for (int k = 1; k < GlobalUtil.NUM_LAYER + 1; k++)         
	          for (int j = 0; j < GlobalUtil.nodes_layer[k]; j++)
	            bh[k][j] = 0.0f;
	        layer_ind = context.getConfiguration().getInt("layer_ind", 0);
	        count=0;
	    }

	    @Override
	    public void reduce(PairOfInts key, Iterable<ModelNode> values, Context context)
	        throws IOException, InterruptedException {  
	      Iterator<ModelNode> iter = values.iterator();

	      // combine
	      reducer_id = key.getLeftElement();
	      while (iter.hasNext()){
	        ModelNode now = iter.next();
	        if (now.getID()<0) {
	            weights = now.getWeight();
	            bh = now.getBH();
	            bv = now.getBV();
	        }
	        else {
	          now.setLayerInd(layer_ind);
	          context.write(key, now);
	          count++;
	        }
	      }
	    }

	    @Override
	    public void cleanup(Context context) throws IOException, InterruptedException {
        modelStructure.setID(-1);
        modelStructure.setLayerInd(layer_ind);
        modelStructure.setBH(bh);
        modelStructure.setBV(bv);
        modelStructure.setWeight(weights);        
	      modelID_Type.set(reducer_id, -1);
	      context.write(modelID_Type, modelStructure);
	      Log.info("!!!!! "+reducer_id+" "+count);
	    }
	  }

	

	  
	  
	  
	  
	protected static class MyMapper extends Mapper<PairOfInts, ModelNode, PairOfInts, ModelNode> {
		@Override
		public void map(PairOfInts key, ModelNode value, Context context)
				throws IOException, InterruptedException {
		  context.write(key, value);
		}
	}



	protected static class MyReducer_Train extends Reducer<PairOfInts, ModelNode, PairOfInts, ModelNode> {
		private static float[][] weights = new float[GlobalUtil.NUM_LAYER+1][]; //space storing the updating weights (first is not used)
		private static float[][] bh = new float[GlobalUtil.NUM_LAYER+1][]; // hidden layer biases (rbm)
		private static float[][] bv = new float[GlobalUtil.NUM_LAYER+1][]; // visible layer biases (rbm)  

		private static int NUM_LAYER = GlobalUtil.NUM_LAYER;
		private static int NODES_INPUT = GlobalUtil.NODES_INPUT;    
		private static final int[] nodes_layer = GlobalUtil.nodes_layer;

		private static float yita_w = GlobalUtil.yita_w, yita_bv = GlobalUtil.yita_bv, yita_bh = GlobalUtil.yita_bh,
				yita_wt = GlobalUtil.yita_wt, yita_bvt = GlobalUtil.yita_bvt, yita_bht = GlobalUtil.yita_bht; // learning rates
		private static float mu = GlobalUtil.mu, reg = GlobalUtil.reg;

    private static final PairOfInts modelID_Type = new PairOfInts();
    private static final ModelNode modelStructure = new ModelNode();   
    private static final ModelNode modelData = new ModelNode();
    private static int reducer_id = 0;
    private static int count =0;
    private static int layer_ind=0;
    private static float[] inc_w;
    private static float[] inc_bv;
    private static float[] inc_bh;
    
    public void setup(Context context) throws IOException{
        layer_ind = context.getConfiguration().getInt("layer_ind", 0);
        count=0;
        inc_w = new float[nodes_layer[layer_ind-1]*nodes_layer[layer_ind]]; // previous increase of weights
        inc_bv = new float[nodes_layer[layer_ind-1]];
        inc_bh = new float[nodes_layer[layer_ind]];
        Arrays.fill(inc_w,0);
        Arrays.fill(inc_bv, 0);
        Arrays.fill(inc_bh, 0);

    }


		@Override
		public void reduce(PairOfInts key, Iterable<ModelNode> values, Context context)
				throws IOException, InterruptedException {  
			Iterator<ModelNode> iter = values.iterator();

      reducer_id = key.getLeftElement();
      while (iter.hasNext()){
        ModelNode now = iter.next();
        if (now.getID()<0) {
            weights = now.getWeight();
            bh = now.getBH();
            bv = now.getBV();
        }
        else 
        if (now.getID()==0){
          work_update(now.getData(), layer_ind);
          context.write(key, now);
          count++;
        }
        else {
          context.write(key, now);
        }
      }
    }

    @Override
    public void cleanup(Context context) throws IOException, InterruptedException {
      modelStructure.setID(-1);
      modelStructure.setLayerInd(layer_ind);
      modelStructure.setBH(bh);
      modelStructure.setBV(bv);
      modelStructure.setWeight(weights);        
      modelID_Type.set(reducer_id, -1);
      context.write(modelID_Type, modelStructure);

      Log.info("!!!!! "+reducer_id+" "+count);
      
      if (layer_ind==GlobalUtil.NUM_LAYER) {
          String dstName= context.getConfiguration().get("sidepath")+"model"+reducer_id;
          try {
            FileSystem fs = FileSystem.get(context.getConfiguration());   
            fs.delete(new Path(dstName),true);
            FSDataOutputStream modelfile=fs.create(new Path(dstName));
            modelStructure.write(modelfile);       
            modelfile.flush();
            modelfile.close();
          }catch (IOException e){
            e.printStackTrace();
          }
      }
    }

    void work_update(float[] data, int layer_ind){
      float[] x0 = new float[nodes_layer[layer_ind - 1]]; // data
      float[] h0 = new float[nodes_layer[layer_ind]];  // hidden
      float[] x1 = new float[nodes_layer[layer_ind - 1]];
      float[] h1 = new float[nodes_layer[layer_ind]];
      //float[] res = new float[nodes_layer[layer_ind]];

      for (int i = 0; i < nodes_layer[layer_ind - 1]; i++)
        x0[i] = data[i];
      
      GlobalUtil.sigm(h0, bh[layer_ind], weights[layer_ind], x0,
            nodes_layer[layer_ind], nodes_layer[layer_ind-1], true);// up sampling

      //for (int j = 0; j < nodes_layer[layer_ind]; j++)
      //    res[j] = h0[j];

      GlobalUtil.sigm(x1, bv[layer_ind], weights[layer_ind], h0,
            nodes_layer[layer_ind], nodes_layer[layer_ind-1], false);// down sampling

      GlobalUtil.sigm(h1, bh[layer_ind], weights[layer_ind], x1,
            nodes_layer[layer_ind], nodes_layer[layer_ind-1], true);

        for (int j = 0; j < nodes_layer[layer_ind]; j++)
          for (int i = 0; i < nodes_layer[layer_ind-1]; i++) {
            inc_w[j*nodes_layer[layer_ind-1] + i] = mu * inc_w[j*nodes_layer[layer_ind-1] + i]
                + yita_w * (h0[j]*x0[i] - h1[j]*x1[i] - reg * weights[layer_ind][j*nodes_layer[layer_ind-1] + i]);
            weights[layer_ind][j*nodes_layer[layer_ind-1] + i] =
                weights[layer_ind][j*nodes_layer[layer_ind-1] + i]
                    +inc_w[j*nodes_layer[layer_ind-1] + i];
          }

        for (int j = 0; j < nodes_layer[layer_ind]; j++) {
          inc_bh[j] = mu * inc_bh[j] + yita_bh*(h0[j] - h1[j] - reg * bh[layer_ind][j]);
          bh[layer_ind][j] = bh[layer_ind][j] + inc_bh[j];
        }

        for (int i = 0; i < nodes_layer[layer_ind-1]; i++) {
          inc_bv[i] = mu * inc_bv[i] + yita_bv*(x0[i] - x1[i] - reg * bv[layer_ind][i]);
          bv[layer_ind][i] = bv[layer_ind][i] + inc_bv[i];
        }
        // print the layer input data (just for testing)
      }    
	}

	




  protected static class MyReducer_GenData extends Reducer<PairOfInts, ModelNode, PairOfInts, ModelNode> {
	    private static float[][] weights = new float[GlobalUtil.NUM_LAYER+1][]; //space storing the updating weights (first is not used)
	    private static float[][] bh = new float[GlobalUtil.NUM_LAYER+1][]; // hidden layer biases (rbm)
	    private static float[][] bv = new float[GlobalUtil.NUM_LAYER+1][]; // visible layer biases (rbm)  

	    private static int NUM_LAYER = GlobalUtil.NUM_LAYER;
	    private static int NODES_INPUT = GlobalUtil.NODES_INPUT;    
	    private static final int[] nodes_layer = GlobalUtil.nodes_layer;


	    private static final PairOfInts modelID_Type = new PairOfInts();
	    private static final ModelNode modelStructure = new ModelNode();   
	    private static final ModelNode modelData = new ModelNode();
	    private static int layer_ind=0;
	    private static int count = 0;
	    
	    
	    public void setup(Context context) throws IOException{
	        layer_ind = context.getConfiguration().getInt("layer_ind", 0);
	        count=0;
	    }


	    @Override
	    public void reduce(PairOfInts key, Iterable<ModelNode> values, Context context)
	        throws IOException, InterruptedException {  
	      Iterator<ModelNode> iter = values.iterator();

	      while (iter.hasNext()){
	        ModelNode now = iter.next();
	        if (now.getID()<0) {
	            weights = now.getWeight();
	            bh = now.getBH();
	            bv = now.getBV();
	            if (layer_ind!=GlobalUtil.NUM_LAYER) 
	                context.write(key, now);
	        }
	        else {
	          if (layer_ind!=GlobalUtil.NUM_LAYER || now.getID()>0) {
	            float[] res = 
	                work_test(now.getData(), layer_ind);
	            now.setLayerInd(layer_ind);
	            now.setData(res);
	            context.write(key, now);
	            count++;
	          }
	        }
	      }
	    }

	    float[] work_test(float[] data, int layer_ind){
	      float[] x0 = new float[nodes_layer[layer_ind - 1]]; // data
	      float[] h0 = new float[nodes_layer[layer_ind]];  // hidden
	      float[] res = new float[nodes_layer[layer_ind]];

	      for (int i = 0; i < nodes_layer[layer_ind - 1]; i++)
	        x0[i] = data[i];
	      
	      GlobalUtil.sigm(h0, bh[layer_ind], weights[layer_ind], x0,
	            nodes_layer[layer_ind], nodes_layer[layer_ind-1], true);// up sampling

	      for (int j = 0; j < nodes_layer[layer_ind]; j++)
	          res[j] = h0[j];
	      return res;
	     }    
	    
  }


  
  
  protected static class MyMapper_Super extends Mapper<PairOfInts, ModelNode, PairOfInts, ModelNode> {
    private static final PairOfInts dataID_ReducerID = new PairOfInts();

    
    @Override
    public void map(PairOfInts key, ModelNode value, Context context)
      throws IOException, InterruptedException {
        if (key.getRightElement()<=0) {
          for(;;) Log.info("!!!!!!!!!!");
        }
        dataID_ReducerID.set(key.getRightElement(), key.getLeftElement());
        context.write(key, value);
    }
  }
  
  protected static class MyReducer_Super extends Reducer<PairOfInts, ModelNode, NullWritable, NullWritable> {
    private static SuperModel superModel = null;
    
    private static int count = 0;
    private static int numModel = 0;
    private static int prevID = -1;
    private static int nEach = GlobalUtil.nodes_layer[GlobalUtil.NUM_LAYER];
    private static float[] layer0;
    public void setup(Context context) throws IOException{
      numModel = context.getConfiguration().getInt("num_reduce_task", 1);
      count = 0;
      prevID = -1;
      layer0 = new float[numModel*nEach];
      superModel = new SuperModel(numModel);
    }


    @Override
    public void reduce(PairOfInts key, Iterable<ModelNode> values, Context context)
        throws IOException, InterruptedException {  
      Iterator<ModelNode> iter = values.iterator();

      if (prevID!=key.getLeftElement()) {
          if (prevID!=-1) {
              superModel.train(layer0);
              count++;
          }
          prevID=key.getLeftElement();
      }
      int modelID = key.getRightElement();
      
      while (iter.hasNext()){
        ModelNode now = iter.next();
        float[] d = now.getData();
        for (int k=0; k<nEach;k++) 
          layer0[modelID*nEach+k] = d[k];
      }
    }

    @Override
    public void cleanup(Context context) {
      //output model in sidedata
      Log.info("!!!!!totoal: "+count);
      String dstName= context.getConfiguration().get("sidepath")+"super";
      try {
        FileSystem fs = FileSystem.get(context.getConfiguration());   
        fs.delete(new Path(dstName),true);
        FSDataOutputStream modelfile=fs.create(new Path(dstName));
        superModel.write(modelfile);       
        modelfile.flush();
        modelfile.close();
      }catch (IOException e){
        e.printStackTrace();
      }
    }
  }

  
  
	
	
	protected static class MyPartitioner extends Partitioner<PairOfInts, ModelNode> {
		@Override
		public int getPartition(PairOfInts key, ModelNode value, int numReduceTasks) {
			return (key.getLeftElement()) % numReduceTasks;
		}
	}


	public AutoCoder(){}

	private static final String INPUT = "input";
	private static final String OUTPUT = "output";
	private static final String NUM_REDUCERS = "numReducers";


	private static int printUsage() {
		System.out.println("usage: [input-path] [output-path] [num-reducers]");
		ToolRunner.printGenericCommandUsage(System.out);
		return -1;
	}

	/**
	 * Runs this tool.
	 */
	@SuppressWarnings({ "static-access" })
	public int run(String[] args) throws Exception {
		Options options = new Options();

		options.addOption(OptionBuilder.withArgName("path").hasArg()
				.withDescription("input path").create(INPUT));
		options.addOption(OptionBuilder.withArgName("path").hasArg()
				.withDescription("output path").create(OUTPUT));
		options.addOption(OptionBuilder.withArgName("num").hasArg()
				.withDescription("number of reducers").create(NUM_REDUCERS));

		CommandLine cmdline;
		CommandLineParser parser = new GnuParser();

		try {
			cmdline = parser.parse(options, args);
		} catch (ParseException exp) {
			System.err.println("Error parsing command line: " + exp.getMessage());
			return -1;
		}

		/*if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)) {
			System.out.println("args: " + Arrays.toString(args));
			HelpFormatter formatter = new HelpFormatter();
			formatter.setWidth(120);
			formatter.printHelp(this.getClass().getName(), options);
			ToolRunner.printGenericCommandUsage(System.out);
			return -1;
		}*/

		//String inputPath = cmdline.getOptionValue(INPUT);
		//String outputPath = cmdline.getOptionValue(OUTPUT);

    String inputPath = "qiwang321/MNIST-mingled-key/part*";
    String outputPath = "shangfu/layeroutput";
    
		int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ?
				Integer.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

				LOG.info("Tool: " + AutoCoder.class.getSimpleName());
				LOG.info(" - input path: " + inputPath);
				LOG.info(" - output path: " + outputPath);
				LOG.info(" - number of reducers: " + reduceTasks);
				Configuration conf = getConf();

	      conf.setInt("num_reduce_task", reduceTasks);
        conf.set("sidepath", outputPath+"_side/");       
        
				
		    Job job0 = Job.getInstance(conf);
        job0.setJobName(AutoCoder.class.getSimpleName());
        job0.setJarByClass(AutoCoder.class);
        job0.setNumReduceTasks(reduceTasks);
        
        job0.getConfiguration().setInt("layer_ind", 0);       
        
    
        FileInputFormat.setInputPaths(job0, new Path(inputPath));
        FileOutputFormat.setOutputPath(job0, new Path(outputPath+"_0"));
    
        job0.setInputFormatClass(KeyValueTextInputFormat.class);
        job0.setOutputFormatClass(SequenceFileOutputFormat.class);

        job0.setMapOutputKeyClass(PairOfInts.class);
        job0.setMapOutputValueClass(ModelNode.class);
        job0.setOutputKeyClass(PairOfInts.class);
        job0.setOutputValueClass(ModelNode.class);

        job0.setMapperClass(MyMapper0.class);
        job0.setReducerClass(MyReducer0.class);
        job0.setPartitionerClass(MyPartitioner.class);
    // Delete the output directory if it exists already.
        Path outputDir = new Path(outputPath+"_0");
        FileSystem.get(getConf()).delete(outputDir, true);
        long startTime = System.currentTimeMillis();
        long codeStart = System.currentTimeMillis();
        double codeTimeSum = 0;
        job0.waitForCompletion(true);
        LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");        
        codeTimeSum+=(System.currentTimeMillis() - startTime) / 1000.0;

				
				
				
				for (int iterations=1; iterations<GlobalUtil.NUM_LAYER+1; iterations++) {
				    Job job1 = Job.getInstance(conf);
				    job1.setJobName(AutoCoder.class.getSimpleName());
				    job1.setJarByClass(AutoCoder.class);
				    job1.setNumReduceTasks(reduceTasks);
				    job1.getConfiguration().setInt("layer_ind", iterations);       
				    FileInputFormat.setInputPaths(job1, new Path(outputPath+"_"+(iterations-1)));
				    FileOutputFormat.setOutputPath(job1, new Path(outputPath+"_"+iterations+"_train"));
				    
		        LOG.info("Tool: " + AutoCoder.class.getSimpleName());
		        LOG.info(" - input path: " + outputPath+"_"+(iterations-1));
		        LOG.info(" - output path: " + outputPath+"_"+iterations+"_train");
		        LOG.info(" - number of reducers: " + reduceTasks);

				
				    job1.setInputFormatClass(SequenceFileInputFormat.class);
				    job1.setOutputFormatClass(SequenceFileOutputFormat.class);

				    job1.setMapOutputKeyClass(PairOfInts.class);
				    job1.setMapOutputValueClass(ModelNode.class);
				    job1.setOutputKeyClass(PairOfInts.class);
				    job1.setOutputValueClass(ModelNode.class);

				    job1.setMapperClass(MyMapper.class);
				    job1.setReducerClass(MyReducer_Train.class);
				    job1.setPartitionerClass(MyPartitioner.class);
				// Delete the output directory if it exists already.
				    outputDir = new Path(outputPath+"_"+iterations+"_train");
				    FileSystem.get(getConf()).delete(outputDir, true);
				    startTime = System.currentTimeMillis();
				    job1.waitForCompletion(true);
				    LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");        
		        codeTimeSum+=(System.currentTimeMillis() - startTime) / 1000.0;

				    
				    
				    
				    
            Job job2 = Job.getInstance(conf);
            job2.setJobName(AutoCoder.class.getSimpleName());
            job2.setJarByClass(AutoCoder.class);
            job2.setNumReduceTasks(reduceTasks);
            job2.getConfiguration().setInt("layer_ind", iterations);       
            FileInputFormat.setInputPaths(job2, new Path(outputPath+"_"+(iterations+"_train")));
            FileOutputFormat.setOutputPath(job2, new Path(outputPath+"_"+iterations));

            LOG.info("Tool: " + AutoCoder.class.getSimpleName());
            LOG.info(" - input path: " + outputPath+"_"+iterations+"_train");
            LOG.info(" - output path: " + outputPath+"_"+iterations);
            LOG.info(" - number of reducers: " + reduceTasks);

            
            job2.setInputFormatClass(SequenceFileInputFormat.class);
            job2.setOutputFormatClass(SequenceFileOutputFormat.class);

            job2.setMapOutputKeyClass(PairOfInts.class);
            job2.setMapOutputValueClass(ModelNode.class);
            job2.setOutputKeyClass(PairOfInts.class);
            job2.setOutputValueClass(ModelNode.class);

            job2.setMapperClass(MyMapper.class);
            job2.setReducerClass(MyReducer_GenData.class);
            job2.setPartitionerClass(MyPartitioner.class);
        // Delete the output directory if it exists already.
            outputDir = new Path(outputPath+"_"+iterations);
            FileSystem.get(getConf()).delete(outputDir, true);
            startTime = System.currentTimeMillis();
            job2.waitForCompletion(true);
            LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");        
            codeTimeSum+=(System.currentTimeMillis() - startTime) / 1000.0;

				}				

				
				
        LOG.info(" - input path: " + outputPath+"_"+GlobalUtil.NUM_LAYER);
        LOG.info(" - output path: " + outputPath);
        reduceTasks = 1;
        LOG.info(" - number of reducers: " + reduceTasks);

        Job job_super = Job.getInstance(conf);
        job_super.setJobName(AutoCoder.class.getSimpleName());
        job_super.setJarByClass(AutoCoder.class);
        job_super.setNumReduceTasks(reduceTasks);

        FileInputFormat.setInputPaths(job_super, new Path(outputPath+"_"+GlobalUtil.NUM_LAYER));
        FileOutputFormat.setOutputPath(job_super, new Path(outputPath));

        job_super.setInputFormatClass(SequenceFileInputFormat.class);
        job_super.setOutputFormatClass(SequenceFileOutputFormat.class);


        job_super.setMapOutputKeyClass(PairOfInts.class);
        job_super.setMapOutputValueClass(ModelNode.class);
        job_super.setOutputKeyClass(NullWritable.class);
        job_super.setOutputValueClass(NullWritable.class);

        job_super.setMapperClass(MyMapper_Super.class);
        job_super.setReducerClass(MyReducer_Super.class);
        job_super.setPartitionerClass(MyPartitioner.class);


        // Delete the output directory if it exists already.
        outputDir = new Path(outputPath);
        FileSystem.get(getConf()).delete(outputDir, true);


        startTime = System.currentTimeMillis();
        job_super.waitForCompletion(true);
        LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");        
        codeTimeSum+=(System.currentTimeMillis() - startTime) / 1000.0;

        Log.info("Final Time: "+((System.currentTimeMillis() - codeStart) / 1000.0)+" seconds,  "+codeTimeSum+" seconds.");
        //prepareNextIteration(inputPath0, outputPath,iterations,conf,reduceTasks);

				
				return 0;
	}

	/**
	 * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
	 */
	public static void main(String[] args) throws Exception {
		ToolRunner.run(new AutoCoder(),args);
	}

}

