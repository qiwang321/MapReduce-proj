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

import org.apache.log4j.Logger;

import cern.colt.Arrays;

import org.uncommons.maths.random.PoissonGenerator;

public class largeSample_v2 extends Configured implements Tool {
  private static final Logger LOG = Logger.getLogger(largeSample_v2.class);

  // Mapper: emits (token, 1) for every word occurrence.
  private static class MyMapper extends Mapper<LongWritable, Text, PairOfInts, Text> {

    // Reuse objects to save overhead of object creation.
    private final static PairOfInts TARGET = new PairOfInts();
    private static float lambda; // parameter for poisson distribution
    private static Random rg; // poisson random number generator
    private static int m, n, km;

    @Override
    public void setup(Context context) {
    	Configuration conf = context.getConfiguration();
    	n = conf.getInt("N", 0);
    	m = conf.getInt("M", 0);
    	km = conf.getInt("KM", 0);
    	lambda = (float)km / (float)m / (float) n; // parameter of the poisson distribution
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
	context.write(TARGET, value);
	rem = rem - n;
	}
	if (rg.nextFloat() > (float)rem/n) return;
	int t = rg.nextInt(m);
	int sec = rg.nextInt();
	  TARGET.set(t, sec);
	context.write(TARGET, value);
    }
  }

  // Reducer: indentity reducer
  private static class MyReducer extends Reducer<PairOfInts, Text, NullWritable, Text> {
	  
	  @Override
	  public void reduce(PairOfInts key, Iterable<Text> values, Context context)
			  throws IOException, InterruptedException {
		  Iterator<Text> iter = values.iterator();
		  while (iter.hasNext()) {
			  context.write(NullWritable.get(), iter.next());
		  }
	  }
  }

  /**
   * Creates an instance of this tool.
   */
  public largeSample_v2() {}

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
    String outputPath = args[0];
    int km = 2;
    int n = 5;
    int reduceTasks = 10;
    LOG.info("Tool: " + largeSample_v2.class.getSimpleName());
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
    job.setJobName(largeSample_v2.class.getSimpleName());
    job.setJarByClass(largeSample_v2.class);

    job.setNumReduceTasks(reduceTasks);

    FileInputFormat.setInputPaths(job, new Path(inputPath));
    FileOutputFormat.setOutputPath(job, new Path(outputPath));
    FileInputFormat.setMinInputSplitSize(job, 660*1024*1024);

    job.setMapOutputKeyClass(PairOfInts.class);
    job.setMapOutputValueClass(Text.class);
    job.setOutputKeyClass(NullWritable.class);
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
  	args = new String[1];
  	for (int i = 0; i < 2; i++) {
  		args[0] = "best5-mingled-small" + i;
    ToolRunner.run(new largeSample_v2(), args);
  	}
  }
}
  
  
  
