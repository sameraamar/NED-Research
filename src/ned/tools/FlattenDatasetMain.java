package ned.tools;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPInputStream;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import ned.main.ExecutorMonitorThread;
import ned.types.Document;
import ned.types.Entry;
import ned.types.GlobalData;
import ned.types.RedisBasedMap;
import ned.types.SerializeHelperAdapter;
import ned.types.Session;
import ned.types.Utility;

public class FlattenDatasetMain {
	private static final String VERSION = "V1";
	private static FlattenToCSVExecutor mapper;
	public static int processed;
	private static Hashtable<String, String> positive;
	private static RedisBasedMap<String, Entry> id2group;
	//private static Hashtable<String, Integer> positive;
	//private static HashtableRedis<Entry> id2group;
	
	public static void main(String[] args) throws IOException
	{
		try {
			
			long base = System.nanoTime();
			String suffex = "5m";
			String folder = "C://data/Thesis/threads_petrovic_all/"; 
			String subfolder = "analysis_3m"; //petrovic_only"; mt_results
			String csvFull = folder+"/"+subfolder+"/dataset_full_"+suffex+"_" + VERSION + ".txt";
			String csvShort = folder+"/"+subfolder+"/dataset_short_"+suffex+"_" + VERSION + ".txt";

			//String filename = "C:/data/events_db/petrovic/relevance_judgments_00000000";
			//flattenLabeledData(filename, "c:/temp/relevance_judgments_00000000_"+suffex+"_" + VERSION + ".csv");
			
			loadLabeledData(folder+"/"+subfolder+"/positive_labeled.txt");

			System.out.println("output to " + csvFull);
			PrintStream shortOut = new PrintStream(new FileOutputStream(csvShort));
			PrintStream fullOut = new PrintStream(new FileOutputStream(csvFull));
			id2group = new RedisBasedMap<String, Entry>("mapper.id2group", false, false, true, new SerializeHelperAdapter<Entry>());

			mapper = new FlattenToCSVExecutor(fullOut, shortOut, id2group, positive, GlobalData.getInstance().getParams().number_of_threads);
			ExecutorMonitorThread monitor = new FlattenDatasetMonitor(mapper.getExecutor(), 2);
			monitor.start();
			doMain();
			
			mapper.shutdown();
			
			monitor.shutdown();
			
			fullOut.close();
			shortOut.close();
			
			String time = Utility.humanTime(TimeUnit.NANOSECONDS.toSeconds(System.nanoTime()-base));
			System.out.println("Finished in " + time + ".");
			
		} catch(Exception e) {
			e.printStackTrace();
		} finally {
			
		}


	}

	public static void doMain() throws IOException {
		GlobalData gd = GlobalData.getInstance();
		
		
		String folder = "C:/data/Thesis/events_db/petrovic";
		//folder = "C:/temp";
		//String[] files = {"my_tweets.txt.gz"};
		String[] files = {"petrovic_00000000.gz",
	                    "petrovic_00500000.gz",
	                    "petrovic_01000000.gz",
	                    "petrovic_01500000.gz",
	                    "petrovic_02000000.gz",
	                    "petrovic_02500000.gz",
	                    "petrovic_03000000.gz",
	                    "petrovic_03500000.gz",
	                    "petrovic_04000000.gz",
	                    "petrovic_04500000.gz",
	                    "petrovic_05000000.gz",
	                    "petrovic_05500000.gz",
	                    "petrovic_06000000.gz",
	                    "petrovic_06500000.gz",
	                    "petrovic_07000000.gz",
	                    "petrovic_07500000.gz",
	                    "petrovic_08000000.gz",
	                    "petrovic_08500000.gz",
	                    "petrovic_09000000.gz",
	                    "petrovic_09500000.gz",
	                    "petrovic_10000000.gz",
	                    "petrovic_10500000.gz",
	                    "petrovic_11000000.gz",
	                    "petrovic_11500000.gz",
	                    "petrovic_12000000.gz",
	                    "petrovic_12500000.gz",
	                    "petrovic_13000000.gz",
	                    "petrovic_13500000.gz",
	                    "petrovic_14000000.gz",
	                    "petrovic_14500000.gz",
	                    "petrovic_15000000.gz",
	                    "petrovic_15500000.gz",
	                    "petrovic_16000000.gz",
	                    "petrovic_16500000.gz",
	                    "petrovic_17000000.gz",
	                    "petrovic_17500000.gz",
	                    "petrovic_18000000.gz",
	                    "petrovic_18500000.gz",
	                    "petrovic_19000000.gz",
	                    "petrovic_19500000.gz",
	                    "petrovic_20000000.gz",
	                    "petrovic_20500000.gz",
	                    "petrovic_21000000.gz",
	                    "petrovic_21500000.gz",
	                    "petrovic_22000000.gz",
	                    "petrovic_22500000.gz",
	                    "petrovic_23000000.gz",
	                    "petrovic_23500000.gz",
	                    "petrovic_24000000.gz",
	                    "petrovic_24500000.gz",
	                    "petrovic_25000000.gz",
	                    "petrovic_25500000.gz",
	                    "petrovic_26000000.gz",
	                    "petrovic_26500000.gz",
	                    "petrovic_27000000.gz",
	                    "petrovic_27500000.gz",
	                    "petrovic_28000000.gz",
	                    "petrovic_28500000.gz",
	                    "petrovic_29000000.gz",
	                    "petrovic_29500000.gz"  
	                   };

		processed = 0;
		int middle_processed = 0;
		int cursor = 0;
		boolean stop = false;
		long base = System.nanoTime();
		long middletime = base;
		//id2group.reset();
		//ids = new ArrayList<String>();
		
    	Session.getInstance().message(Session.ERROR, "Reader", "Loading data...");

    	int offset = gd.getParams().offset;
		int skip_files = (offset / 500_000);
		offset = offset % 500_000;
		int fileidx = -1;

    	for (String filename : files) {
			fileidx++;
			if (stop)
				break;
			
			if(fileidx < skip_files)
			{
            	Session.getInstance().message(Session.INFO, "Reader", "Skipping file " + fileidx + ": " + filename);
				continue;
			}
	    	Session.getInstance().message(Session.INFO, "Reader", "reading from file: " + filename);

			
			GZIPInputStream stream = new GZIPInputStream(new FileInputStream(folder + "/" + filename));
			Reader decoder = new InputStreamReader(stream, "UTF-8");
			BufferedReader buffered = new BufferedReader(decoder);
			
			String line=buffered.readLine();
			while(!stop && line != null)
	        {
				cursor += 1;
				if(cursor < offset)
				{
					if(cursor % 50000 == 0)
						System.out.println("Cursor " + cursor);
					
					line=buffered.readLine();
					continue;					
				}
				
				//GlobalData.getInstance().setDocumentFromRedis("id2doc_parser", doc.getId(), doc);
				mapper.submit(line);
								
	            processed ++;

	            if (processed % (gd.getParams().print_limit) == 0)
	            {
	        		long tmp = System.nanoTime() - middletime;
	            	double average2 = 1.0 * TimeUnit.NANOSECONDS.toMillis(tmp) / middle_processed;
	            	average2 = Math.round(100.0 * average2) / 100.0;
	            	
	            	StringBuffer msg = new StringBuffer();
	            	msg.append("> Processed " ).append ( processed ).append(" docs. ");
	            	long seconds = TimeUnit.NANOSECONDS.toSeconds( System.nanoTime() - base);
	            	msg.append(" elapsed time: ").append(Utility.humanTime(seconds));
	            	//msg.append("(AHT: ").append(average2).append(" ms). ");
	            	
	            	Session.getInstance().message(Session.INFO, "Reader", msg.toString());
	            	
            		middletime = System.nanoTime();
            		middle_processed = 0;
	            }
	            
	            if (processed % (gd.getParams().print_limit*50) == 0)
	            {
	            	System.out.println("wait for executor to clean the list...");
	            	while(mapper.counter.get() < processed)
	            	{
	            		try {
							Thread.sleep(30);
						} catch (InterruptedException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
	            	}
	            	System.out.println("clear memory...");
	            	id2group.clear();
	            }
	            
	            if (processed >= gd.getParams().max_documents)
	            	stop = true;
	            
	            line=buffered.readLine();
	        }
			buffered.close();
	        
		}
		
		long current = System.nanoTime();

		long seconds = TimeUnit.NANOSECONDS.toSeconds(current-base);
		Session.getInstance().message(Session.INFO, "Summary", "Done in " + Utility.humanTime(seconds) );
	}
	
	public static void loadLabeledData(String inputfile) throws IOException
	{
		positive = new Hashtable<String, String> ();
		FileInputStream stream = new FileInputStream(inputfile);
		Reader decoder = new InputStreamReader(stream, "UTF-8");
		BufferedReader buffered = new BufferedReader(decoder);
		System.out.println("Loading file: " + inputfile + " to memory");
		
		String line=buffered.readLine(); //header
		line=buffered.readLine();
		while(line!=null)
		{
			String[] values = line.split("\t");
			positive.put(values[0], values[2]);
			line=buffered.readLine();
		}
		
		buffered.close();
	}
	/*
	public static void flattenAndLoadLabeledData(String inputfile, String outputfile) throws IOException
	{		
		FileInputStream stream = new FileInputStream(inputfile);
		Reader decoder = new InputStreamReader(stream, "UTF-8");
		BufferedReader buffered = new BufferedReader(decoder);
		System.out.println("Loading file: " + inputfile + " to memory");
		
		PrintStream output = new PrintStream(new FileOutputStream(outputfile));
		positive = new Hashtable<String, Integer> ();
		System.out.println("Flatten file: " + inputfile + " to " + outputfile);
		
		
		String line=buffered.readLine();
		int count = 0;
		int failed = 0;
		int skip = 0;
		while(line != null)
        {
			JsonParser jsonParser = new JsonParser();
			JsonObject jsonObj = jsonParser.parse(line).getAsJsonObject();
			Integer topic = jsonObj.get("topic_id").getAsInt();
			String status = jsonObj.get("status").getAsString();
			
			//JsonElement tweet = jsonObj.get("json");
			//if(tweet==null || tweet.isJsonNull())
			//{
			//	skip++;
			//	line=buffered.readLine();
			//	continue;
			//}
			
			String id = jsonObj.get("_id").toString();
			//String json = jsonObj.get("json").toString();
			//Document doc = Document.parse(json, false);
			
			count ++;
			
			StringBuffer sb = new StringBuffer();
			sb.append(id).append(",");
			sb.append(topic).append(",");
			sb.append(status);
			
			sb.append("\n");
			positive.put(id, topic);
			
			output.print(sb);
			
			if(count % 1000 == 0)
				System.out.println("Loaded " + count);
			
			line=buffered.readLine();
			
        }
		
		buffered.close();
		output.close();
		System.out.println("File loaded: "+count+" successful, "+ failed +" failed, "+ skip +" skipped.");
		
	}

	public static Integer getTopic(String id) 
	{
		return positive.getOrDefault(id, -1);
	}
	*/
}
