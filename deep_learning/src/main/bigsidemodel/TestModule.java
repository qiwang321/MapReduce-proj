package bigsidemodel;





import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.Arrays;
import java.util.Random;
import java.util.Vector;

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
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;


public class TestModule{

  //modify
  private static int nModel = 10;
  private static SuperModel sm;
  private static Vector<ModelNode> models = new Vector<ModelNode>();
  
  @SuppressWarnings("deprecation")
  public static void initial(String path) throws IOException {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    FSDataInputStream modelfile;
    models.clear();
    for (int i=0;i<nModel;i++){
      modelfile=FileSystem.get(conf).open(new Path(path+"model"+i));
      ModelNode now = new ModelNode();
      now.readFields(modelfile);
      models.add(now);
    }
    modelfile=FileSystem.get(conf).open(new Path(path+"super"));
    sm = new SuperModel(0);
    sm.readFields(modelfile);
    System.out.println(sm.nEach+"  "+sm.nModel+"  "+"  "+sm.NODES_INPUT);    
  }
  
  public static float[] test(float[] test_records){
    return sm.test(test_records);
  }
  
  
  public static void test(Configuration conf, String inputPath, String outputPath) throws IOException{
    FSDataInputStream testfile=FileSystem.get(conf).open(new Path(inputPath));
    BufferedReader reader = new BufferedReader(new InputStreamReader(testfile));
    FSDataOutputStream testoutput=FileSystem.get(conf).create(new Path(outputPath));
    BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(testoutput));
    float[] test_records = new float[GlobalUtil.NODES_INPUT];
    float[] result;
    float[] smInput = new float[sm.NODES_INPUT];
    while (reader.ready()){
   // for(int k=0; k < 100;k++) {
      String line = reader.readLine();
      if (line.length() == 0)
      	continue;
      String[] items = line.trim().split("\\s+");
      for (int i=0;i<GlobalUtil.NODES_INPUT;i++) 
          test_records[i]=Float.parseFloat(items[i]) / 255.0f;
      
      float[] tmp;
      for (int i=0;i<nModel;i++){
          tmp = models.get(i).sim(test_records);
          for (int j=0;j<sm.nEach;j++)
              smInput[i*sm.nEach+j]=tmp[j];
      }
      result = test(smInput);
      
      for (int j = 0; j < result.length; j++)
      	writer.write(result[j] + " ");
      writer.write("\n");
      
    }
    writer.close();
    reader.close();
  }
  
    public static final String[] testfile = {"qiwang321/best5-56x56/part1-56x56", "qiwang321/best5-56x56/part5-56x56", "qiwang321/best5-56x56/part7-56x56", "qiwang321/best5-56x56/part9-56x56", "qiwang321/best5-56x56/part10-56x56"};
  public static void main(String[] args) throws IOException{
      initial("shangfu/bigoutput_side/");
      for (int i = 0; i < 5; i++) {
	  test(new Configuration() , testfile[i], "shangfu/big_test_out_side/class" + i);
      	System.out.println("tested set " + i);
      }
  }

}
