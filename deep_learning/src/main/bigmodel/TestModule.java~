package model;





import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
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

  private static SuperModel sm = new SuperModel();
  
  @SuppressWarnings("deprecation")
  public static void initial(String path) throws IOException {
    Configuration conf = new Configuration();
    FileSystem fs = FileSystem.get(conf);
    SequenceFile.Reader reader = null;
    try {
        reader = new SequenceFile.Reader(fs, new Path(path), conf);
        NullWritable key = NullWritable.get();
            ReflectionUtils.newInstance(reader.getKeyClass(), conf);
        SuperModel value = new SuperModel();
            ReflectionUtils.newInstance(reader.getValueClass(), conf);
        long position = reader.getPosition();
        /*while (reader.next(key, value)) {
            String syncSeen = reader.syncSeen() ? "*" : "";
            System.out.printf("[%s%s]\t%s\t%s\n", position, syncSeen, key, value);
            position = reader.getPosition(); // beginning of next record
        }*/
        if (reader.next(key, value)) {
           sm=value; 
        }
    } finally {
        IOUtils.closeStream(reader);
    }
  }
  
  public static float[] test(float[] test_records){
    return sm.test(test_records);
  }
  
  
  public static void test(Configuration conf, String inputPath, String outputPath) throws IOException{
    FSDataInputStream testfile=FileSystem.get(conf).open(new Path(inputPath));
    BufferedReader reader = new BufferedReader(new InputStreamReader(testfile));
    FSDataOutputStream testoutput=FileSystem.get(conf).create(new Path(outputPath));
    BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(testoutput));
    float[] test_records = new float[GlobalUtil.sizeInput];
    float[] result;
    while (reader.ready()){
    //for(int k=0; k < 100;k++) {
      String line = reader.readLine();
      if (line.length() == 0)
      	continue;
      String[] items = line.trim().split("\\s+");
      for (int i=0;i<GlobalUtil.sizeInput;i++) 
          test_records[i]=Float.parseFloat(items[i]) / 255.0f;
      result = test(test_records);
      
      for (int j = 0; j < result.length; j++)
      	writer.write(result[j] + " ");
      writer.write("\n");
      
    }
    writer.close();
    reader.close();
  }
  
  
  public static void main(String[] args) throws IOException{
  	String modelPath = args[0];
    initial(modelPath + "/part-r-00000");
    String dataPath = args[1];
    String outputPath = args[2];
    for (int i = 1; i <= 5; i++) {
    	test(new Configuration(), dataPath + "/part" + i, 
    			outputPath + "/class" + i);
    	System.out.println("tested set " + i);
    }
  }

}
