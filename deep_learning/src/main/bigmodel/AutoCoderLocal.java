package bigmodel;
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

public class AutoCoderLocal extends Configured implements Tool {
	private static final Logger LOG = Logger.getLogger(AutoCoderLocal.class);

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
		private static int NODES_INPUT = GlobalUtil.NODES_INPUT; // 27*27    
		private static final int[] train_len = GlobalUtil.train_len; 
		private static final int[] test_len = GlobalUtil.test_len;
		private static final int[] nodes_layer = GlobalUtil.nodes_layer;
		
		private static final int window = GlobalUtil.window;
		//private static float[] field;
		private static float[] weights_field;
		private static float[] bh_field;
		private static float[] bv_field;
		private static final int sizeInput = GlobalUtil.sizeInput;
		private static final int numWindows = GlobalUtil.numWindows;
		private static final int imageSize = GlobalUtil.imageSize;
		private static final int step = GlobalUtil.step;
		private static final int raw2d = GlobalUtil.raw2d;
		private static final int new2d = GlobalUtil.new2d;


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
					weights[k][j] = 0.01f * (float)rd.nextGaussian();


			for (int k = 1; k < GlobalUtil.NUM_LAYER + 1; k++) 
				for (int j = 0; j < GlobalUtil.nodes_layer[k-1]; j++)
					bv[k][j] = 0.0f;

			for (int k = 1; k < GlobalUtil.NUM_LAYER + 1; k++)         
				for (int j = 0; j < GlobalUtil.nodes_layer[k]; j++)
					bh[k][j] = 0.0f;

			//reader.close();
			//cluster.close();
			weights_field = new float[NODES_INPUT * window * window];
			bh_field = new float[NODES_INPUT];
			bv_field = new float[sizeInput];
		
			for (int j = 0; j < NODES_INPUT * window * window; j++)
				weights_field[j] = 0.01f * (float)rd.nextGaussian();
			for (int j = 0; j < NODES_INPUT; j++)
				bh_field[j] = 0.0f;
			for (int j = 0; j < sizeInput; j++)
				bv_field[j] = 0.0f;
			
			for (int k = 1; k < GlobalUtil.NUM_LAYER + 1; k++) { 
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
			StringTokenizer itr = new StringTokenizer(line);
			float[] data=new float[sizeInput];

			int tot=0;
			while (itr.hasMoreTokens()){
				String curr = itr.nextToken();
				data[tot] = Float.parseFloat(curr)/255.0f;
				tot++;
			}

			
			input_update(data); // layer of local receptive field
			for (int i=1; i <= GlobalUtil.NUM_LAYER;i++) {
				layer_ind = i;
				work_update();
			}
			num_train_data++;
		}

		public void cleanup(Context context) throws IOException, InterruptedException {
			comp.set(num_train_data);

			model.setWeight(weights, weights_field);
			model.setBV(bv, bv_field);
			model.setBH(bh, bh_field);
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

				/*for (int i = 0; i < nodes_layer[layer_ind]; i++) {
					if (rd.nextFloat() < h0[i])
						h0[i] = 1;
					else
						h0[i] = 0;
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
		
		void input_update(float[] data) {
			float[] x0 = new float[sizeInput]; // data
			float[] h0 = new float[NODES_INPUT];  // hidden
			float[] x1 = new float[sizeInput];
			float[] h1 = new float[NODES_INPUT];
			float[] inc_w = new float[NODES_INPUT * window * window]; // previous increase of weights
			float[] inc_bv = new float[sizeInput];
			float[] inc_bh = new float[NODES_INPUT];
			Arrays.fill(inc_w,0);
			Arrays.fill(inc_bv, 0);
			Arrays.fill(inc_bh, 0);
			//up sampling
		/*	for (int l = 0; l < 3; l++) 
			for (int i = 0; i < numWindows; i++)
				for (int j = 0; j < numWindows; j++) {
				int ind_h = l*new2d + i*numWindows + j;
				h0[ind_h] = -bh_field[ind_h];
				for (int s = 0; s < window; s++)
					for (int t = 0; t < window; t++) {
						int ind_v = l*raw2d + (i*step+s)*imageSize + j*step+t;
						int ind_w = ind_h*window*window + s*window+t;
						h0[ind_h] = h0[ind_h] - weights_field[ind_w] * data[ind_v];
					}
				h0[ind_h] = 1.0f / (1.0f + (float)Math.exp(h0[ind_h]));
			}
			*/
			
			for (int l = 0; l < 3; l++) 
			for (int i = 0; i < numWindows; i++)
				for (int j = 0; j < numWindows; j++) {
				int ind_h = l*new2d + i*numWindows + j;
				h0[ind_h] = 0;
				for (int s = 0; s < window; s++)
					for (int t = 0; t < window; t++) {
						int ind_v = l*raw2d + (i*step+s)*imageSize + j*step+t;
						int ind_w = ind_h*window*window + s*window+t;
						h0[ind_h] = h0[ind_h] + data[ind_v];
					}
				h0[ind_h] /= window*window;
			}
			

			for (int j = 0; j < NODES_INPUT; j++)
				sample_mem[0][j] = h0[j];


		/*	// down sampling	
			for (int l = 0; l < 3; l++) 
			for (int i = 0; i < numWindows; i++)
				for (int j = 0; j < numWindows; j++) {
				int ind_h = l*new2d + i*numWindows + j;
				h0[ind_h] = -bh_field[ind_h];
				for (int s = 0; s < window; s++)
					for (int t = 0; t < window; t++) {
						int ind_v = l*raw2d + (i*step+s)*imageSize + j*step+t;
						int ind_w = ind_h*window*window + s*window+t;
						x1[ind_v] = x1[ind_v] - weights_field[ind_w] * h0[ind_h];
					}
			}
			
			
			for (int d = 0; d < sizeInput; d++) {
				x1[d] = 1.0f / (1.0f + (float)Math.exp(x1[d]-bv_field[d]));
			}
			
			//up sampling
			for (int l = 0; l < 3; l++) 
			for (int i = 0; i < numWindows; i++)
				for (int j = 0; j < numWindows; j++) {
				int ind_h = l*new2d + i*numWindows + j;
				h1[ind_h] = -bh_field[ind_h];
				for (int s = 0; s < window; s++)
					for (int t = 0; t < window; t++) {
						int ind_v = l*raw2d + (i*step+s)*imageSize + j*step+t;
						int ind_w = ind_h*window*window + s*window+t;
						h1[ind_h] = h1[ind_h] - weights_field[ind_w] * data[ind_v];
					}
				h1[ind_h] = 1.0f / (1.0f + (float)Math.exp(h1[ind_h]));
			}
			

			for (int l = 0; l < 3; l++) 
			for (int i = 0; i < numWindows; i++)
				for (int j = 0; j < numWindows; j++) {
					
					int ind_h = l*new2d + i*numWindows + j;
					
					for (int s = 0; s < window; s++)
						for (int t = 0; t < window; t++) {
							int ind_v = l*raw2d + (i*step+s)*imageSize + j*step+t;
							int ind_w = ind_h*window*window + s*window+t;
							inc_w[ind_w] = mu * inc_w[ind_w] + yita_wt * 
									(h0[ind_h]*data[ind_v] - h1[ind_h]*x1[ind_v] - reg * weights_field[ind_w]);
							weights_field[ind_w] = weights_field[ind_w] + inc_w[ind_w];
						}
				}

			for (int i = 0; i < NODES_INPUT; i++) {
				inc_bh[i] = mu * inc_bh[i] + yita_bht*(h0[i] - h1[i] - reg * bh_field[i]);
				bh_field[i] = bh_field[i] + inc_bh[i];
			}

			for (int i = 0; i < sizeInput; i++) {
				inc_bv[i] = mu * inc_bv[i] + yita_bht*(data[i] - x1[i] - reg * bv_field[i]);
				bv_field[i] = bv_field[i] + inc_bv[i];
			} */
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

			// Initialize the  memory for weight parameters
			for (int k = 1; k <= GlobalUtil.NUM_LAYER; k++) {
				weights[k] = new float[GlobalUtil.nodes_layer[k-1] * GlobalUtil.nodes_layer[k]];
				bv[k] = new float[GlobalUtil.nodes_layer[k-1]];
				bh[k] = new float[GlobalUtil.nodes_layer[k]];
			}
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
		void combine(ModelNode1 model, ModelNode1 now) {
			if (count==0) {
				model = now;
			}
			count++;
		}
		 */
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
				System.out.println("writing successful!!!");
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				System.out.println("Reduce write error1 !!!!!!!!!!!!!!!!");
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				System.out.println("Reduce write error2 !!!!!!!!!!!!!!!!!!!");
			}
		}		


	}

	protected static class MyPartitioner extends Partitioner<Text, PairOfStrings> {
		@Override
		public int getPartition(Text key, PairOfStrings value, int numReduceTasks) {
			return (0) % numReduceTasks;
		}
	}


	public AutoCoderLocal(){}

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

		String inputPath = cmdline.getOptionValue(INPUT) + "/part-r-00000";
		String outputPath = cmdline.getOptionValue(OUTPUT);
		String dataPath = cmdline.getOptionValue(INPUT) + "/common";
		//String inputPath = "/home/qiwang321/mapreduce-data/data/in-mingled1-5/part*";
		//String outputPath = "output";
		//String dataPath = "/home/qiwang321/mapreduce-data/data/in-mingled1-5/common";
		int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ?
				Integer.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

				LOG.info("Tool: " + AutoCoderLocal.class.getSimpleName());
				LOG.info(" - input path: " + inputPath);
				LOG.info(" - output path: " + outputPath);
				LOG.info(" - number of reducers: " + reduceTasks);
				Configuration conf = getConf();
				initialParameters(conf);

				conf.set("dataPath", dataPath);

				Job job = Job.getInstance(conf);
				job.setJobName(AutoCoderLocal.class.getSimpleName());
				job.setJarByClass(AutoCoderLocal.class);
				// set the path of the information of k clusters in this iteration
				job.getConfiguration().set("sidepath", inputPath+"/side_output");       
				job.setNumReduceTasks(reduceTasks);
				

				dataShuffle();

				FileInputFormat.setInputPaths(job, new Path(inputPath));
				FileOutputFormat.setOutputPath(job, new Path(outputPath));
				FileInputFormat.setMinInputSplitSize(job, 1000*1024*1024);
				FileInputFormat.setMaxInputSplitSize(job, 1000*1024*1024);
				
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
		ToolRunner.run(new AutoCoderLocal(),args);
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
