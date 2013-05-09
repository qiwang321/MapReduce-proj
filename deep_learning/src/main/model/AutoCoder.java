package model;
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
import edu.umd.cloud9.io.pair.PairOfStrings;

public class AutoCoder extends Configured implements Tool {
	private static final Logger LOG = Logger.getLogger(AutoCoder.class);

	protected static class MyMapper extends Mapper<LongWritable, Text, IntWritable, ModelNode> {
		private static final IntWritable comp = new IntWritable();
		private static final ModelNode model = new ModelNode();    
		private static int num_train_data = 0;

		private static final Random rd = new Random();  
		private static float[][] sample_mem = new float[GlobalUtil.NUM_LAYER+1][]; //space storing the MCMC samples
		private static float[][] weights = new float[GlobalUtil.NUM_LAYER+1][]; //space storing the updating weights (first is not used)
		private static float[][] bh = new float[GlobalUtil.NUM_LAYER+1][]; // hidden layer biases (rbm)
		private static float[][] bv = new float[GlobalUtil.NUM_LAYER+1][]; // visible layer biases (rbm)  

		private static int NUM_LAYER = GlobalUtil.NUM_LAYER;
		private static int NODES_INPUT = GlobalUtil.NODES_INPUT;    
		private static final int[] train_len = GlobalUtil.train_len; 
		private static final int[] test_len = GlobalUtil.test_len;
		private static final int[] nodes_layer = GlobalUtil.nodes_layer;



		private static float yita_w = GlobalUtil.yita_w, yita_bv = GlobalUtil.yita_bv, yita_bh = GlobalUtil.yita_bh,
				yita_wt = GlobalUtil.yita_wt, yita_bvt = GlobalUtil.yita_bvt, yita_bht = GlobalUtil.yita_bht; // learning rates
		private static float mu = GlobalUtil.mu, reg = GlobalUtil.reg;

		private static int layer_ind=0;
		
		float[][] inc_w = new float[NUM_LAYER+1][]; // previous increase of weights
		float[][] inc_bv = new float[NUM_LAYER+1][];
		float[][] inc_bh = new float[NUM_LAYER+1][];


		private static float read_float(BufferedReader reader) throws NumberFormatException, IOException{
			while (reader.ready()){        
				String line=reader.readLine();
				if (line.length()==0) continue;
				return Float.parseFloat(line);
			}
			return 0;
		}

		public void setup(Context context) throws IOException{
			// load the information of k clusters 
			//String file = context.getConfiguration().get("sidepath");
			//FSDataInputStream cluster=FileSystem.get(context.getConfiguration()).open(new Path(file));
			//BufferedReader reader = new BufferedReader(new InputStreamReader(cluster));


			// Initialize the  memory for MCMC samples
			for (int k = 0; k < GlobalUtil.NUM_LAYER + 1; k++) {
				sample_mem[k] = new float[GlobalUtil.nodes_layer[k]];          
			}


			// Initialize the  memory for weight parameters
			for (int k = 1; k < GlobalUtil.NUM_LAYER + 1; k++) {
				weights[k] = new float[GlobalUtil.nodes_layer[k-1] * GlobalUtil.nodes_layer[k]];
				bv[k] = new float[GlobalUtil.nodes_layer[k-1]];
				bh[k] = new float[GlobalUtil.nodes_layer[k]];
			}


			/*      for (int k = 0; k < GlobalUtil.NUM_LAYER + 1; k++) 
        for (int j = 0; j < GlobalUtil.nodes_layer[k]; j++)
          sample_mem[k][j]=read_float(reader);
			 */    

			for (int k = 1; k < GlobalUtil.NUM_LAYER + 1; k++) 
				for (int j = 0; j < GlobalUtil.nodes_layer[k-1] * GlobalUtil.nodes_layer[k]; j++)
					weights[k][j] = 0.1f * (float)rd.nextGaussian();


			for (int k = 1; k < GlobalUtil.NUM_LAYER + 1; k++) 
				for (int j = 0; j < GlobalUtil.nodes_layer[k-1]; j++)
					bv[k][j] = 0.0f;

			for (int k = 1; k < GlobalUtil.NUM_LAYER + 1; k++)         
				for (int j = 0; j < GlobalUtil.nodes_layer[k]; j++)
					bh[k][j] = 0.0f;

			//reader.close();
			//cluster.close();

			for (int k = 1; k <= NUM_LAYER; k++) {
				inc_w[k] = new float[nodes_layer[k-1]*nodes_layer[k]]; // previous increase of weights
				inc_bv[k] = new float[nodes_layer[k-1]];
				inc_bh[k] = new float[nodes_layer[k]];
				
				Arrays.fill(inc_w[k],0);
				Arrays.fill(inc_bv[k], 0);
				Arrays.fill(inc_bh[k], 0);
			}
			
			num_train_data = 0;
		}


		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			String line = value.toString();
			if (line.length() == 0) return;
			StringTokenizer itr = new StringTokenizer(line);
			float[] data=new float[NODES_INPUT];

			int tot=0;
			while (itr.hasMoreTokens()){
				String curr = itr.nextToken();
				data[tot] = Float.parseFloat(curr) / 255.0f;
				tot++;
			}

			for (int i=0; i <nodes_layer[0];i++) {
				sample_mem[0][i] = data[i];
			}
			for (int i=1; i <= GlobalUtil.NUM_LAYER;i++) {
				layer_ind = i;
				work_update();
			}
			num_train_data++;
		}

		public void cleanup(Context context) throws IOException, InterruptedException {
			comp.set(num_train_data);

//			ArrayListOfFloatsWritable[] W = new ArrayListOfFloatsWritable[GlobalUtil.NUM_LAYER+1];
//			ArrayListOfFloatsWritable[] BV = new ArrayListOfFloatsWritable[GlobalUtil.NUM_LAYER+1];
//			ArrayListOfFloatsWritable[] BH = new ArrayListOfFloatsWritable[GlobalUtil.NUM_LAYER+1];

/*			for (int k = 1; k < GlobalUtil.NUM_LAYER + 1; k++) {
				W[k] = new ArrayListOfFloatsWritable(weights[k]);
				BV[k] = new ArrayListOfFloatsWritable(bv[k]);
				BH[k] = new ArrayListOfFloatsWritable(bh[k]);
			}
*/
			model.setWeight(weights);
			model.setBV(bv);
			model.setBH(bh);
			context.write(comp, model);
		}

		void work_update(){
			float[] x0 = new float[nodes_layer[layer_ind - 1]]; // data
			float[] h0 = new float[nodes_layer[layer_ind]];  // hidden
			float[] x1 = new float[nodes_layer[layer_ind - 1]];
			float[] h1 = new float[nodes_layer[layer_ind]];



			for (int i = 0; i < nodes_layer[layer_ind - 1]; i++)
				x0[i] = sample_mem[layer_ind - 1][i];

			//if (layer_ind != NUM_LAYER) 
			{ // normal layer        

				//perform real computation
				GlobalUtil.sigm(h0, bh[layer_ind], weights[layer_ind], x0,
						nodes_layer[layer_ind], nodes_layer[layer_ind-1], true);// up sampling

				for (int j = 0; j < nodes_layer[layer_ind]; j++)
					sample_mem[layer_ind][j] = h0[j];

/*				for (int i = 0; i < nodes_layer[layer_ind]; i++) {
					if (rd.nextFloat() < h0[i])
						h0[i] = 1.0f;
					else
						h0[i] = 0.0f;
				}*/


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
			/*else { // top layer
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

				for (int j = 0; j < nodes_layer[NUM_LAYER]; j++)
					for (int i = 0; i < nodes_layer[NUM_LAYER-1]; i++) {
						inc_w[j*nodes_layer[NUM_LAYER-1] + i] = mu * inc_w[j*nodes_layer[NUM_LAYER-1] + i]
								+ yita_wt * (h0[j]*x0[i] - h1[j]*x1[i] - reg * weights[NUM_LAYER][j*nodes_layer[NUM_LAYER-1] + i]);
						weights[NUM_LAYER][j*nodes_layer[NUM_LAYER-1] + i] =
								weights[NUM_LAYER][j*nodes_layer[NUM_LAYER-1] + i]
										+inc_w[j*nodes_layer[NUM_LAYER-1] + i];
					}

				for (int j = 0; j < nodes_layer[NUM_LAYER]; j++) {
					inc_bh[j] = mu * inc_bh[j] + yita_bht*(h0[j] - h1[j] - reg * bh[NUM_LAYER][j]);
					bh[NUM_LAYER][j] = bh[NUM_LAYER][j] + inc_bh[j];
				}

				for (int i = 0; i < nodes_layer[NUM_LAYER-1]; i++) {
					inc_bv[i] = mu * inc_bv[i] + yita_bvt*(x0[i] - x1[i] - reg * bv[NUM_LAYER][i]);
					bv[NUM_LAYER][i] = bv[NUM_LAYER][i] + inc_bv[i];
				}
				// print the layer input data (just for testing)
			}*/
		}
	}



	protected static class MyReducer extends Reducer<IntWritable, ModelNode, NullWritable, SuperModel> {
		private static final Text result = new Text();
		private static ArrayList<ModelNode> modelSet = new ArrayList<ModelNode>();     

		private static final Random rd = new Random();  
		private static float[][] weights = new float[GlobalUtil.NUM_LAYER+1][]; //space storing the updating weights (first is not used)
		private static float[][] bh = new float[GlobalUtil.NUM_LAYER+1][]; // hidden layer biases (rbm)
		private static float[][] bv = new float[GlobalUtil.NUM_LAYER+1][]; // visible layer biases (rbm)  

		private static int NUM_LAYER = GlobalUtil.NUM_LAYER;
		private static int NODES_INPUT = GlobalUtil.NODES_INPUT;    
		private static final int[] train_len = GlobalUtil.train_len; 
		private static final int[] test_len = GlobalUtil.test_len;
		private static final int[] nodes_layer = GlobalUtil.nodes_layer;

		private static float yita_w = GlobalUtil.yita_w, yita_bv = GlobalUtil.yita_bv, yita_bh = GlobalUtil.yita_bh,
				yita_wt = GlobalUtil.yita_wt, yita_bvt = GlobalUtil.yita_bvt, yita_bht = GlobalUtil.yita_bht; // learning rates
		private static float mu = GlobalUtil.mu, reg = GlobalUtil.reg;
		private static int layer_ind=0;
		String dataPath;

		private static int count = 0;

		public void setup(Context context) throws IOException{
			// load the information of k clusters 
			layer_ind = context.getConfiguration().getInt("layer_ind", 0);

			count = 0;
			dataPath = context.getConfiguration().get("dataPath");
		}

		/*    
    public void cleanup(Context context) throws IOException, InterruptedException {
      result.set("result");
      context.write(result, model);
  }
		 */


		@Override
		public void reduce(IntWritable key, Iterable<ModelNode> values, Context context)
				throws IOException, InterruptedException {  
			Iterator<ModelNode> iter = values.iterator();

			// combine
			while (iter.hasNext()){
				ModelNode now = iter.next();
				//combine(model,now);
				modelSet.add(now);
			}

		}

		/*
		void combine(ModelNode model, ModelNode now) {
			if (count==0) {
				model = now;
			}
			count++;
		}
		 */
		@Override
		public void cleanup(Context context) {
			SuperModel consensus = new SuperModel(modelSet);

			// train
			Path inputPath = new Path(dataPath);
			Configuration conf = context.getConfiguration();
			FileSystem fs;
			try {
				fs = FileSystem.get(conf);
				//FSDataInputStream din = new FSDataInputStream(fs.open(inputPath));
				BufferedReader in = new BufferedReader(new InputStreamReader(fs.open(inputPath)));
				consensus.train(in);
				context.write(NullWritable.get(), consensus);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

	}

	protected static class MyPartitioner extends Partitioner<Text, PairOfStrings> {
		@Override
		public int getPartition(Text key, PairOfStrings value, int numReduceTasks) {
			return (0) % numReduceTasks;
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

		if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)) {
			System.out.println("args: " + Arrays.toString(args));
			HelpFormatter formatter = new HelpFormatter();
			formatter.setWidth(120);
			formatter.printHelp(this.getClass().getName(), options);
			ToolRunner.printGenericCommandUsage(System.out);
			return -1;
		}

		String inputPath = cmdline.getOptionValue(INPUT) + "/part*";
		String outputPath = cmdline.getOptionValue(OUTPUT);
		//String inputPath = "mingled_v2/part*";
		//String outputPath = "output";
		String dataPath = cmdline.getOptionValue(INPUT) + "/common";
		int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ?
				Integer.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

				LOG.info("Tool: " + AutoCoder.class.getSimpleName());
				LOG.info(" - input path: " + inputPath);
				LOG.info(" - output path: " + outputPath);
				LOG.info(" - number of reducers: " + reduceTasks);
				Configuration conf = getConf();
				initialParameters(conf);

				conf.set("dataPath", dataPath);

				conf.set("mapreduce.map.memory.mb", "2048");
  				conf.set("mapreduce.map.java.opts", "-Xmx2048m");
    				conf.set("mapreduce.reduce.memory.mb", "2048");
    				conf.set("mapreduce.reduce.java.opts", "-Xmx2048m");

				Job job = Job.getInstance(conf);
				job.setJobName(AutoCoder.class.getSimpleName());
				job.setJarByClass(AutoCoder.class);
				// set the path of the information of k clusters in this iteration
				job.getConfiguration().set("sidepath", inputPath+"/side_output");       
				job.setNumReduceTasks(reduceTasks);
				

				dataShuffle();

				FileInputFormat.setInputPaths(job, new Path(inputPath));
				FileOutputFormat.setOutputPath(job, new Path(outputPath));
				FileInputFormat.setMaxInputSplitSize(job, 1000*1024*1024);
				FileInputFormat.setMinInputSplitSize(job, 1000*1024*1024);
								

				job.setInputFormatClass(TextInputFormat.class);
				job.setOutputFormatClass(SequenceFileOutputFormat.class);

				job.setMapOutputKeyClass(IntWritable.class);
				job.setMapOutputValueClass(ModelNode.class);
				job.setOutputKeyClass(NullWritable.class);
				job.setOutputValueClass(SuperModel.class);

				job.setMapperClass(MyMapper.class);
				job.setReducerClass(MyReducer.class);
				job.setPartitionerClass(MyPartitioner.class);


				// Delete the output directory if it exists already.
				Path outputDir = new Path(outputPath);
				FileSystem.get(getConf()).delete(outputDir, true);


				long startTime = System.currentTimeMillis();
				job.waitForCompletion(true);
				LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");        

				//prepareNextIteration(inputPath0, outputPath,iterations,conf,reduceTasks);
				
				return 0;
	}

	/**
	 * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
	 */
	public static void main(String[] args) throws Exception {
		ToolRunner.run(new AutoCoder(),args);
	}


	public static void initialParameters(Configuration conf){

	}
	public static void dataShuffle(){

	}
}

	/*
	public static void prepareNextIteration(String input, String output, int iterations, Configuration conf, int reduceTasks){
		String dstName= input+"/cluster"+iterations;
		try {
			FileSystem fs = FileSystem.get(conf);   
			fs.delete(new Path(dstName),true);
			FSDataOutputStream clusterfile=fs.create(new Path(dstName));

			for (int i=0;i<reduceTasks;i++){
				String srcName= output+"/part-r-"+String.format("%05d", i);
				FSDataInputStream cluster=fs.open(new Path(srcName));
				BufferedReader reader = new BufferedReader(new InputStreamReader(cluster));
				while (reader.ready()){
					String line=reader.readLine()+"\n";
					if (line.length()>5)
						clusterfile.write(line.getBytes());
				}
				reader.close();
				cluster.close(); 
			}
			clusterfile.flush();
			clusterfile.close();
		}catch (IOException e){
			e.printStackTrace();
		}
	}
}
	 */
