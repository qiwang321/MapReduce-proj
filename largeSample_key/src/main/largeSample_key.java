import java.io.IOException;
import java.util.Iterator;
import java.util.Random;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import edu.umd.cloud9.io.pair.PairOfInts;
import edu.umd.cloud9.io.pair.PairOfWritables;

import org.apache.log4j.Logger;

import cern.colt.Arrays;

public class largeSample_key extends Configured implements Tool {
	private static final Logger LOG = Logger.getLogger(largeSample_key.class);
	private static int unikey; 

	// Mapper: emits (token, 1) for every word occurrence.
	private static class MyMapper extends Mapper<LongWritable, Text, PairOfInts, PairOfWritables<IntWritable, Text>> {

		// Reuse objects to save overhead of object creation.
		private final static PairOfInts TARGET = new PairOfInts();
		private static Random rg; // poisson random number generator
		private static int m, n, km;
		private final static IntWritable seckey = new IntWritable(); 
		private final static PairOfWritables<IntWritable, Text> sendv = new PairOfWritables<IntWritable, Text>();

		@Override
		public void setup(Context context) {
			Configuration conf = context.getConfiguration();
			n = conf.getInt("N", 0);
			m = conf.getInt("M", 0);
			km = conf.getInt("KM", 0);
			rg = new Random();
		}
		@Override
		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			if(value.getLength() == 0) return;
			int rem = km;
			for (int i = 0; i < km/n; i++) {
				int t = rg.nextInt(m);
				int sec = rg.nextInt();
				TARGET.set(t, sec);
				
				if (rg.nextFloat() > 1.0f/11.0f) 
					seckey.set(0);
				else {
					unikey++;
					seckey.set(unikey);
				}
				sendv.set(seckey, value);

				context.write(TARGET, sendv);
				rem = rem - n;
			}
			
			if (rg.nextFloat() > (float)rem/n) return;
			
			int t = rg.nextInt(m);
			int sec = rg.nextInt();
			TARGET.set(t, sec);

			if (rg.nextFloat() > 1.0f/11.0f) 
				seckey.set(0);
			else {
				unikey++;
				seckey.set(unikey);
			}
			sendv.set(seckey, value);

			context.write(TARGET, sendv);
		}
	}

	// Reducer: indentity reducer
	private static class MyReducer extends Reducer<PairOfInts, PairOfWritables<IntWritable, Text>, IntWritable, Text> {

		private static PairOfWritables<IntWritable, Text> tmp;
		@Override
		public void reduce(PairOfInts key, Iterable<PairOfWritables<IntWritable, Text>> values, Context context)
				throws IOException, InterruptedException {
			Iterator<PairOfWritables<IntWritable, Text>> iter = values.iterator();
			while (iter.hasNext()) {
				tmp = iter.next();
				context.write(tmp.getKey(), tmp.getValue());
			}
		}
	}

	/**
	 * Creates an instance of this tool.
	 */
	public largeSample_key() {}

	private static final String INPUT = "input";
	private static final String OUTPUT = "output";
	private static final String NUM_PARTITIONS = "numPartitions"; // desired number of data partitions (equal to the number of reducers)
	private static final String KM = "KM"; // total resampled data wanted
	private static final String N = "N"; // total number of records

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
				.withDescription("number of partitions").create(NUM_PARTITIONS));
		options.addOption(OptionBuilder.withArgName("km").hasArg()
				.withDescription("number desired samples").create(KM));
		options.addOption(OptionBuilder.withArgName("n").hasArg()
				.withDescription("total number of records").create(N));

		CommandLine cmdline;
		CommandLineParser parser = new GnuParser();

		try {
			cmdline = parser.parse(options, args);
		} catch (ParseException exp) {
			System.err.println("Error parsing command line: " + exp.getMessage());
			return -1;
		}

		/*    if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)||
    		!cmdline.hasOption(KM) || !cmdline.hasOption(N)) {
      System.out.println("args: " + Arrays.toString(args));
      HelpFormatter formatter = new HelpFormatter();
      formatter.setWidth(120);
      formatter.printHelp(this.getClass().getName(), options);
      ToolRunner.printGenericCommandUsage(System.out);
      return -1;
    }*/

		String inputPath = "/home/qiwang321/mapreduce-data/data56x56/best5-56x56/part*";
		//String inputPath = "MNIST-mingled-key/part-r-*";
		String outputPath = args[0];
		int km = 2;
		int n = 10;
		int reduceTasks = 10;
		LOG.info("Tool: " + largeSample_key.class.getSimpleName());
		LOG.info(" - input path: " + inputPath);
		LOG.info(" - output path: " + outputPath);
		LOG.info(" - number of partitions: " + reduceTasks);
		LOG.info(" - total number of records: " + n);
		LOG.info(" - desired number of samples: " + km);

		Configuration conf = getConf();
		conf.setInt("KM", km); // desired number of samples
		conf.setInt("N", n); // total number of records
		conf.setInt("M", reduceTasks); // total number of partitions
		Job job = Job.getInstance(conf);
		job.setJobName(largeSample_key.class.getSimpleName());
		job.setJarByClass(largeSample_key.class);

		job.setNumReduceTasks(reduceTasks);

		FileInputFormat.setInputPaths(job, new Path(inputPath));
		FileOutputFormat.setOutputPath(job, new Path(outputPath));
		FileInputFormat.setMinInputSplitSize(job, 660*1024*1024);

		job.setMapOutputKeyClass(PairOfInts.class);
		job.setMapOutputValueClass(PairOfWritables.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(Text.class);

		job.setMapperClass(MyMapper.class);
		//job.setCombinerClass(MyReducer.class);
		job.setReducerClass(MyReducer.class);

		// Delete the output directory if it exists already.
		Path outputDir = new Path(outputPath);
		FileSystem.get(conf).delete(outputDir, true);

		long startTime = System.currentTimeMillis();
		job.waitForCompletion(true);
		LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

		return 0;
	}

	/**
	 * Dispatches command-line arguments to the tool via the {@code ToolRunner}.
	 */
	public static void main(String[] args) throws Exception {
		unikey = 0;
		args = new String[1];
		for (int i = 0; i < 11; i++) {
			args[0] = "best5-mingled-key" + i;
			ToolRunner.run(new largeSample_key(), args);
		}
	}
}


