package ned.graphstream;


import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.Reader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPInputStream;
import org.graphstream.algorithm.ConnectedComponents;
import org.graphstream.graph.Graph;
import org.graphstream.graph.Node;
import org.graphstream.graph.implementations.SingleGraph;

import ned.types.Document;
import ned.types.GlobalData;
import ned.types.Session;
import ned.types.Utility;

public class IdentifyInteractionGraphStream {
	private static final String VERSION = "V1";
	public static int processed;
	//private static Hashtable<String, Integer> positive;
	public static HashMap<String, String> id2parent;
	private static HashMap<String, Integer> idInvertedIndex;
	private static ArrayList<String> idArray;
	private static Graph g;
	
	public static void main(String[] args) throws IOException
	{
		try {
			
			long base = System.nanoTime();
			String suffex = "30m";
			String folder = "C:/data/Thesis/threads_petrovic_all/mt_results"; //petrovic_only"; mt_results; analysis_3m
			//String csvfilename = folder+"/dataset_"+suffex+"_" + VERSION + ".txt";

			g = new SingleGraph("Tutorial 1");
			//g.display();
			id2parent = new HashMap<String, String>();
			doMain();
			dumpGroups(folder+"/id2parent_"+suffex+"_" + VERSION + ".txt");

			ConnectedComponents cc = new ConnectedComponents();
	        cc.init(g);
	        
	        List<Node> nods = cc.getGiantComponent();
	        System.out.println("giant component: " + nods.size());
	        for(Node n : nods)
	        {
	        	//n.addAttribute("ui.color", Color.RED);
	        	n.addAttribute("ui.style", "fill-color: rgb(100,0,0);");
	        	n.addAttribute("ui.style", "fill-color: red;");
	        	//graph.addAttribute(arg0, arg1);
	        }
	        
	        System.out.printf("%d connected component(s) in this graph, so far.%n",
	                cc.getConnectedComponentsCount());
						
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
		idInvertedIndex = new HashMap<String, Integer>();
		idArray = new ArrayList<String>();
		
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
				
				String rply = doc.getReplyTo();
				String rtwt = doc.getRetweetedId();
				String qte = doc.getQuotedStatusId();
				
				String parentId = null;
				int parentType = 0;
				if(rply != null)
				{
					parentId = rply;
					parentType = 1;
				}

				if(rtwt != null)
				{
					parentId = rtwt;
					parentType = 2;
				}

				if(qte != null)
				{
					parentId = qte;
					parentType = 3;
				}
				
				if(!idInvertedIndex.containsKey(doc.getId()))
				{
					idInvertedIndex.put(doc.getId(), idArray.size());
					idArray.add(doc.getId());
				}
				
				g.addNode(doc.getId());
				
				if(parentId != null)
				{
					id2parent.put(doc.getId(), parentId + "," + parentType);
					if(!idInvertedIndex.containsKey(parentId))
					{
						idInvertedIndex.put(parentId, idArray.size());
						idArray.add(parentId);
					}
					if(g.getNode(parentId) == null)
						g.addNode(parentId);
					
					g.addEdge(doc.getId() + "." + parentId, doc.getId(), parentId);
				}
				
	            processed ++;

	            if (processed % (gd.getParams().print_limit) == 0)
	            {
	        		long tmp = System.nanoTime() - middletime;
	            	double average2 = 1.0 * TimeUnit.NANOSECONDS.toMillis(tmp) / middle_processed;
	            	average2 = Math.round(100.0 * average2) / 100.0;
	            	
	            	StringBuffer msg = new StringBuffer();
	            	msg.append("> Processed " ).append ( processed ).append(" docs. ");
	            	msg.append("mem map size: ").append(id2parent.size()).append(". ");
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
	            
//	            if (processed % (gd.getParams().print_limit*50) == 0)
//	            {
//	            	id2parent.save();
//	            }
	            
	            if (processed >= gd.getParams().max_documents)
	            	stop = true;
	            
	            line=buffered.readLine();
	        }
			buffered.close();
	        
		}
    	
    	//id2parent.save();
		
		long current = System.nanoTime();

		long seconds = TimeUnit.NANOSECONDS.toSeconds(current-base);
		Session.getInstance().message(Session.INFO, "Summary", "Done in " + Utility.humanTime(seconds) );
	}
	
	private static void dumpGroups(String filename) throws FileNotFoundException 
	{
		System.out.println("Writing groups to file: " + filename);
		PrintStream groupsOut = new PrintStream(filename);
		
		groupsOut.println("id,parentId,parentType");
		for (int i=0; i<idArray.size(); i++)
		{
			String e = id2parent.get(idArray.get(i));
			
			StringBuffer sb = new StringBuffer();
			sb.append(idArray.get(i));
			sb.append(",");
			sb.append(e);
			
			groupsOut.println(sb.toString());
		}
		groupsOut.close();
	}
	
}
