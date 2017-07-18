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
import ned.types.SerializeHelper;
import ned.types.SerializeHelperAdapterDirtyBit;
import ned.types.SerializeHelperAdapterSimpleType;
import ned.types.Session;
import ned.types.Utility;

public class IdentifyInteractionGraph {
	private static final String VERSION = "V1";
	public static int processed;
	//private static Hashtable<String, Integer> positive;
	public static RedisBasedMap<String, Entry> id2group;
	public static RedisBasedMap<String, String> id2root;
	private static ArrayList<String> ids;
	
	public static void main(String[] args) throws IOException
	{
		try {
			
			long base = System.nanoTime();
			String suffex = "30m";
			String folder = "C:/data/Thesis/threads_petrovic_all/analysis_3m"; //petrovic_only"; mt_results
			//String csvfilename = folder+"/dataset_"+suffex+"_" + VERSION + ".txt";

			//id2group = new Hashtable<String, Entry>(); // HashtableRedis<Entry>("mapper.id2group", Entry.class);
			id2group = new RedisBasedMap<String, Entry>("mapper.id2group", true, false, true, new SerializeHelperAdapterDirtyBit<Entry>());
			id2root = new RedisBasedMap<String, String>("mapper.id2root", true, false, true, new SerializeHelperAdapterSimpleType<String>(String.class));
			doMain();
			dumpGroups(folder+"/id2group_"+suffex+"_" + VERSION + "_bb.txt");
		    
			System.out.println("Finished in " + (TimeUnit.NANOSECONDS.toMillis(System.nanoTime()-base)/1000.0) + " seconds.");
						
		} catch(Exception e) {
			e.printStackTrace();
		} finally {
			
		}


	}

	public static void doMain() throws IOException {
		GlobalData gd = GlobalData.getInstance();
		
		
		String folder = "C:/data/Thesis/events_db/petrovic";
		//folder = "C:\\temp";
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
		ids = new ArrayList<String>();
		
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

				Document doc = Document.parse(line, false);
				if(doc == null)
				{
					line=buffered.readLine();
					continue;
				}
				
				ids.add(doc.getId());
				
				{
					String rply = doc.getReplyTo();
					String rtwt = doc.getRetweetedId();
					String qte = doc.getQuotedStatusId();
					
					String parentId = null, parentType = "";
					
					if(rply != null)
					{
						parentId = rply;
						parentType = "rply";
					}
					else if(rtwt != null)
					{
						parentId = rtwt;
						parentType = "rtwt";
					}
					else if(qte != null)
					{
						parentId = qte;
						parentType = "qte";
					}
						
					
					if(parentId == null)
					{
						Entry e = new Entry(doc.getId(), "", doc.getTimestamp(), 0);
						id2group.put(doc.getId(), e);
						id2root.put(doc.getId(), doc.getId());
					}
					else if(parentId != null)
					{	
						Entry leadE = id2group.get(parentId);
						if (leadE == null) //86370333774462976
						{
							leadE = new Entry(parentId, "", 0, 0, 0);
							id2group.put(parentId, leadE);
							id2root.put(parentId, parentId);
						}

						String tmp = id2root.get(parentId);
						id2root.put(doc.getId(), tmp);
						long firstTimeStamp = id2group.get(tmp).getTimestamp();
						Entry e = new Entry(parentId, parentType, firstTimeStamp, doc.getTimestamp(), leadE.getLevel()+1);
						id2group.put(doc.getId(), e);
					}
					
				}
				
	            processed ++;

	            if (processed % (gd.getParams().print_limit) == 0)
	            {
	        		long tmp = System.nanoTime() - middletime;
	            	double average2 = 1.0 * TimeUnit.NANOSECONDS.toMillis(tmp) / middle_processed;
	            	average2 = Math.round(100.0 * average2) / 100.0;
	            	
	            	StringBuffer msg = new StringBuffer();
	            	msg.append("> Processed " ).append ( processed ).append(" docs. ");
	            	msg.append("mem map size: ").append(id2group.size()).append(". ");
	            	long seconds = TimeUnit.NANOSECONDS.toSeconds( System.nanoTime() - base);
	            	msg.append(" elapsed time: ").append(Utility.humanTime(seconds));
	            	//msg.append("(AHT: ").append(average2).append(" ms). ");
	            	
	          		//for (String id  : id2group.keySet()) {
	          		//	if(!ids.contains(id))
	          		//		msg.append(id + " \n ");
	          		//}
	            	
	            	Session.getInstance().message(Session.INFO, "Reader", msg.toString());
	            	
            		middletime = System.nanoTime();
            		middle_processed = 0;
	            }
	            
	            if (processed % (gd.getParams().print_limit*50) == 0)
	            {
	            	id2group.save();
	            }
	            
	            if (processed >= gd.getParams().max_documents)
	            	stop = true;
	            
	            line=buffered.readLine();
	        }
			buffered.close();
	        
		}
    	
    	id2group.save();
		
		long current = System.nanoTime();

		long seconds = TimeUnit.NANOSECONDS.toSeconds(current-base);
		Session.getInstance().message(Session.INFO, "Summary", "Done in " + Utility.humanTime(seconds) );
	}
	
	private static void dumpGroups(String filename) throws FileNotFoundException 
	{
		System.out.println("Writing groups to file: " + filename);
		PrintStream groupsOut = new PrintStream(filename);
		
		System.out.println("there are " + ids.size() + " ids to plot");
		
		groupsOut.println("id,parentId,parenttype,depth,root_id,timestamp,time_delta");
		int size = ids.size();
		for (int i=0; i<size; i++)
		{
			String currentId = ids.get(i);
			Entry e = id2group.get(currentId);
			
			String parentId = e.getParentId();
			StringBuffer sb = new StringBuffer();
			sb.append(currentId);
			sb.append(",");
			sb.append(parentId);
			sb.append(",");
			sb.append(e.getParentType());
			sb.append(",");
			sb.append(e.getLevel());
			sb.append(",");
			sb.append(id2root.get(currentId));
			sb.append(",");
			sb.append(e.getTimestamp());
			sb.append(",");
			sb.append(e.getTimeDelta());

			groupsOut.println(sb.toString());
		}
		groupsOut.close();
	}
	
}
