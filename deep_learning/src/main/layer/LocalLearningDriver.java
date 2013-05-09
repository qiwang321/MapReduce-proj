package layer;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;


public class LocalLearningDriver{
  private static final Random rd = new Random();  
  private static final String HELP = "help";
  private static final String OUTPUT = "output";
  
  private static final GlobalUtil GB = new GlobalUtil();
//global variables
  //static float[][] sample_mem = new float[GlobalUtil.NUM_LAYER+1][]; //space storing the MCMC samples
  static float[][] weights = new float[GlobalUtil.NUM_LAYER+1][]; //space storing the updating weights (first is not used)
  static float[][] bh = new float[GlobalUtil.NUM_LAYER+1][]; // hidden layer biases (rbm)
  static float[][] bv = new float[GlobalUtil.NUM_LAYER+1][]; // visible layer biases (rbm)

  
//  private static final String input="points_input";

  @SuppressWarnings({ "static-access" })
  public static void main(String[] args) {
    Options options = new Options();

    options.addOption(new Option(HELP, "display help options"));

    options.addOption(OptionBuilder.withArgName("path").hasArg()
        .withDescription("result path").create(OUTPUT));
    
    CommandLine cmdline = null;
    CommandLineParser parser = new GnuParser();

    try {
      cmdline = parser.parse(options, args);
    } catch (ParseException exp) {
      System.err.println("Error parsing command line: " + exp.getMessage());
    }

    if (!cmdline.hasOption(OUTPUT)) {
      System.out.println("args: " + Arrays.toString(args));
      HelpFormatter formatter = new HelpFormatter();
      formatter.setWidth(120);
      formatter.printHelp(LocalLearningDriver.class.getName(), options);
      System.exit(-1);
    }
    
    if (cmdline.hasOption(HELP)) {
      System.out.println("args: " + Arrays.toString(args));
      HelpFormatter formatter = new HelpFormatter();
      formatter.setWidth(120);
      formatter.printHelp(LocalLearningDriver.class.getName(), options);
      System.exit(-1);
    }

    String output = cmdline.getOptionValue(OUTPUT);
    System.out.println("Side output: "+output);

    
    
    // Initialize the  memory for MCMC samples
/*    for (int k = 0; k < GB.NUM_LAYER + 1; k++) {
      sample_mem[k] = new float[GB.nodes_layer[k]];
    
      for (int j = 0; j < GB.nodes_layer[k]; j++)
        sample_mem[k][j] = rd.nextFloat();
    }
*/
    
    // Initialize the  memory for weight parameters
    for (int k = 1; k < GB.NUM_LAYER + 1; k++) {
      weights[k] = new float[GB.nodes_layer[k-1] * GB.nodes_layer[k]];
      bh[k] = new float[GB.nodes_layer[k]];
      bv[k] = new float[GB.nodes_layer[k-1]];

      for (int j = 0; j < GB.nodes_layer[k-1] * GB.nodes_layer[k]; j++)
        weights[k][j] = (float) (0.1 * (rd.nextFloat()* 2 - 1));
      for (int j = 0; j < GB.nodes_layer[k-1]; j++)
        //bv[k][j] = 1.0 / nodes_layer[k-1] * ((float)rand()/RAND_MAX * 2 - 1);
        bv[k][j] = (float) 0.0;
      for (int j = 0; j < GB.nodes_layer[k]; j++)
        //bh[k][j] = 1.0 / nodes_layer[k] * ((float)rand()/RAND_MAX * 2 - 1);
        bh[k][j] = (float) 0.0;
    }

    
    Path outputPoi = new Path(output);
    try {
      FileSystem fs = FileSystem.get(new Configuration());
      fs.delete(outputPoi, true);
      FSDataOutputStream sidefile=fs.create(new Path(output+"/side_output"));
 
    /*  for (int k = 0; k < GB.NUM_LAYER + 1; k++) {
        for (int j = 0; j < GB.nodes_layer[k]; j++)
          sidefile.write((Float.toString(sample_mem[k][j])+"\n").getBytes());
       }
      */
      
      for (int k = 1; k < GB.NUM_LAYER + 1; k++) {
        for (int j = 0; j < GB.nodes_layer[k-1] * GB.nodes_layer[k]; j++)
          sidefile.write((Float.toString(weights[k][j])+"\n").getBytes());
      }
      
      
      for (int k = 1; k < GB.NUM_LAYER + 1; k++) {
        for (int j = 0; j < GB.nodes_layer[k-1]; j++)
          sidefile.write((Float.toString(bv[k][j])+"\n").getBytes());
      }
      
      for (int k = 1; k < GB.NUM_LAYER + 1; k++) {        
        for (int j = 0; j < GB.nodes_layer[k]; j++)
          sidefile.write((Float.toString(bh[k][j])+"\n").getBytes());
      }
        
      sidefile.flush();
      sidefile.close();
      
    }catch (IOException exp){
      exp.printStackTrace();
    }    
  } 
}
