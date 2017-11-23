package ned.tools;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.regex.Pattern;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import ned.types.Document;
import ned.types.GlobalData;

public class EnrichPositiveLabeledDataset {

	private static final String DELIMITER = " ||| ";
	private static int LIMIT = 5000000;

	public static void main(String[] args) throws Exception {
		//getDocuments();
		String folder = "C:/data/Thesis/threads_petrovic_all/analysis_3m";
		String filename = "C:/data/Thesis/threads_petrovic_all/full_all.txt";

		Hashtable<String, String> labeled = new Hashtable<String, String>();
	
		step1_2_LoadLabeled("C:/data/Thesis/threads_petrovic_all/mechanical_turk_positive_labeled.txt", labeled);
		step1_1_loadLabeled("C:/data/Thesis/events_db/petrovic/relevance_judgments_00000000", labeled);
		System.out.println( labeled );

		String yesOut = folder+"/has_topics_YES.txt";
		String noOut = folder+"/has_topics_NO.txt";
		step2_countHits(labeled, filename, yesOut, noOut);
		

		String labeledFileName = folder+"/positive_labeled.txt";
		step3_createLabeledDataset(yesOut, labeledFileName);
		
		//updateText();
	}



	private static void step3_createLabeledDataset(String filename, String labeledFileName) {
		BufferedReader br = null;
	    
	    try {
			br = new BufferedReader(new FileReader(filename));
		    String line = br.readLine();
		    PrintStream out = new PrintStream(new FileOutputStream(labeledFileName));

		    int count = 0;
		    out.print("id\tleadId\ttopic\ttext\n");
		    while (line != null) {
		    	line = br.readLine();
		        if(line == null || line.trim().equals(""))
		        	continue;
		        
		        count++;
		        if(count % 1000 == 0)
		        	System.out.println("createLabeledDataset:\t" + count);
		        
		        String[] values = line.split("\t");
		        
		        String topic = values[8];
		        String id = values[1];
		        String leadId = values[0];
		        String text = values[9];
		        
		    	out.print(id + "\t" + leadId + "\t" + topic + "\t" + text + "\n");
		    	
		    }
		    
		    br.close();
		    out.close();
		    
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
		    
		}
		
	}


	private static Map<String, String> markClustersWithTopics(String filename, Hashtable<String, String> labeled) {
		BufferedReader br = null;
	    HashMap<String, String> map = new HashMap<String, String>();
	    
	    try {
			br = new BufferedReader(new FileReader(filename));
		    String line = br.readLine();

		    int count = 0;
		    
		    while (line != null) {
		    	line = br.readLine();
		        if(line == null || line.trim().equals(""))
		        	continue;
		        
		        count++;
		        if(count % 1000000 == 0)
		        	System.out.println("mark topics:\t" + count + "\t. Found:\t" + map.size() + "\tevent-related items");
		        
		        String[] values = line.split(Pattern.quote(DELIMITER));
		        
		        String topic = labeled.getOrDefault( values[1], "" ); //find the topic of the leader
		    	if(topic.length() > 0)
		    	{
		    		//String tmp = map.get(values[0]); // compare with the topic of the id itself
		    		//if(tmp != null && !tmp.equals(topic))// tmp.indexOf(":" + topic)==-1)
		    		//	topic += ":" + tmp;
		    		
		    		map.put(values[0], topic);
		    	}
		    }
		    br.close();
		    
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
		    
		}
		
		return map;
	}
	


	private static void getDocuments() throws Exception {
		GlobalData.getInstance().getParams().resume_mode = true;
		GlobalData.getInstance().init();
		System.out.println("------------------------------------------");

		String[] ids = {"97829502591320067", "98454114152878081", "97986298270330880", "98199872209035265", "98454114152878081", "98079151780675586",
				"90224059228499969", "90239997587886080", "91813843075993600"};
		
		for (String id : Arrays.asList( ids )) {
			Document doc = GlobalData.getInstance().id2doc.get(id);
			System.out.println(doc==null ? "NULL" : doc.toString());
		}
	}


	static public void updateText()
	{
		String filename = "../temp/threads_50000000_13000000/res_short.txt";
		String filenameOut = "../temp/threads_50000000_13000000/res_001.txt";
		
		
		BufferedReader br = null;
		try {
			PrintStream out = new PrintStream(new FileOutputStream(filenameOut));

			br = new BufferedReader(new FileReader(filename));
		    String line = br.readLine();
		    
		    out.print(line + "\n");

		    int count = 0;
		    int printed = 0;
		    String leadId = null, entropy = null;
		    String size;
	        line = br.readLine();
	        System.out.println(line);
		    while (line != null) {
		        if(line.trim().equals(""))
		        	continue;
		        
		    	StringTokenizer tk = new StringTokenizer(line, DELIMITER);
		    	leadId = tk.nextToken();
		    	entropy = tk.nextToken();
		    	size = tk.nextToken();
		    	size = tk.nextToken();
		    	
		    	Double entr = null;
		    	Integer s = null;
		    	
		        boolean skip = false;
		        try {
		        	entr = Double.parseDouble(entropy);
		        } catch (Exception e)
		        {
		        	e.printStackTrace();
		        	System.out.println(e.getMessage());
		        }

		        try {
		        	s = Integer.parseInt(size);
		        } catch (Exception e)
		        {
		        	e.printStackTrace();
		        	System.out.println(e.getMessage());
		        }

		        if(entr != null && entr < 1.2 )
		        	skip = true;
		        
		        if(s != null && s < 10 )
		        	skip = true;
		        
		        if(!skip)
		        {
		        	out.print(line + "\n");
		        	printed ++;
		        }
		        count++;
		        if(count % 500_000 == 0)
		        	System.out.println("processed: " + count);
		        line = br.readLine();
		    }
		    
		    
		    br.close();
		    System.out.println("count: " + count + " , printed: " + printed );
		    
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
		    
		}
		
	}
	
	

	static public void step2_countHits(Hashtable<String, String> labeled, String clustersInput, String yesOut, String noOut)
	{
		Map<String, String> cluster2Topic = markClustersWithTopics(clustersInput, labeled);
		
		BufferedReader br = null;
		try {
			PrintStream yout = new PrintStream(new FileOutputStream(yesOut));
			PrintStream nout = new PrintStream(new FileOutputStream(noOut));
			//PrintStream allout = new PrintStream(new FileOutputStream(noOut+"all.txt"));

			br = new BufferedReader(new FileReader(clustersInput));
		    String line = br.readLine();
		    
		    nout.print("leadId\tid\ttimestamp\tdistance\tentropy\tusers\tsize\tage\ttopic\ttext\n");
		    yout.print("leadId\tid\ttimestamp\tdistance\tentropy\tusers\tsize\tage\ttopic\ttext\n");
		    //allout.print("topic" + DELIMITER + line + "\n");
			
		    int yesEventCount = 0;
		    int totalLines = 1;
		    String leadId;
	        int clustersCount = 0;
	        
	        ArrayList<String[]> cluster = new ArrayList<String[]>();
	        String currentLeadId = "";
	        String currentTopic = "";
	        
			while (line != null) {
		        line = br.readLine();
		        if(line==null || line.trim().equals(""))
		        {
		        	continue;
		        }
		        //allout.print(line+"\n");
		        totalLines++;

		        String[] values = line.split(Pattern.quote(DELIMITER));
		        leadId = values[0];

		        //1: id
		    	//2: created at
		    	//3: time-stamp
		    	//4: nearest
		    	//5: nearest distance
		    	//6: entropy
		    	//7: users
		    	//8: size
		    	//9: age
		    	//10: score
		    	//11: topic... removed!
		    	//12: text

		        if(leadId.equals(currentLeadId) )
		        {
		        	if(currentTopic == null && cluster2Topic.get(values[1]) != null)
			    	{
			    		currentTopic = cluster2Topic.get(values[1]);
			    	}
		        }
		        
		        else 
		        {
					if(printCluster(yout, nout, cluster, currentTopic))
		        	{
						clustersCount++;
						if(currentTopic!=null && !currentTopic.isEmpty())
				        	yesEventCount++;
		        	}

			        cluster = new ArrayList<String[]>();
			        currentLeadId = leadId;
			        currentTopic = cluster2Topic.get(leadId);
			        
		        }

		        cluster.add( values );

		    	
		        if(totalLines % 500_000 == 0)
		        	System.out.println("Split Event-Realted vs. No-Event-Related: " + totalLines + " lines. " + clustersCount + " clusters. yes: " + yesEventCount + ", no: " + (clustersCount-yesEventCount));
		        
		        if(totalLines % LIMIT == 0)
		       		break;
		        
		    }
		    
			if(printCluster(yout, nout, cluster, currentTopic))
        	{
				clustersCount++;
				if(currentTopic!=null && !currentTopic.isEmpty())
					yesEventCount++;
        	}
			
		    br.close();
		    yout.close();
		    nout.close();
		    //allout.close();
        	System.out.println("Split Event-Realted vs. No-Event-Related: " + totalLines + " lines. yes: " + clustersCount + ", no: " + (clustersCount-yesEventCount));
		    
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
		    
		}
		
	}
	
	
	private static boolean printCluster(PrintStream yout, PrintStream nout, ArrayList<String[]> cluster, String currentTopic) {
		if(cluster.size() == 0)
			return false;
		
		PrintStream out;
		if(currentTopic != null)
    		out = yout;
    	else
    		out = nout;
    	
		for(String[] values : cluster)
		{
	        StringBuilder sb = new StringBuilder();
	        sb.append( values[0] ).append( "\t" );
	        sb.append( values[1] ).append( "\t" );
	        sb.append( values[3] ).append( "\t" );
	        sb.append( values[5] ).append( "\t" );
	        sb.append( values[6] ).append( "\t" );
	        sb.append( values[7] ).append( "\t" );
	        sb.append( values[8] ).append( "\t" );
	        sb.append( values[9] ).append( "\t" );
	        sb.append( currentTopic == null ? "" : currentTopic).append( "\t" );
	        sb.append( values[12] ).append( "\n" );
	        
	        String s = sb.toString();
			
			out.print(s);
		}
		
		return true;
	}



	private static void step1_2_LoadLabeled(String inputfile, Hashtable<String, String> labeled) throws IOException
	{
		FileInputStream stream = new FileInputStream(inputfile);
		Reader decoder = new InputStreamReader(stream, "UTF-8");
		BufferedReader buffered = new BufferedReader(decoder);
		System.out.println("Loading file: " + inputfile + " to memory");
		
		String line=buffered.readLine();
		int count = 0;
		int failed = 0;
		int skip = 0;
		while(line != null)
        {
			if(!line.isEmpty())
			{
				String[] values = line.split("\t");
				String id = values[0];
				String topic = values[1];
				
				count ++;
				
				if(!labeled.contains(id))
					labeled.put(id, topic);
							
				if(count % 1000 == 0)
					System.out.println("Loaded " + count);
			}
			line=buffered.readLine();
			line = line==null ? null : line.trim();
        }
		
		buffered.close();
		System.out.println("File loaded: "+count+" successful, "+ failed +" failed, "+ skip +" skipped.");
	}
	
	private static void step1_1_loadLabeled(String inputfile, Hashtable<String, String> labeled) throws IOException
	{		
		FileInputStream stream = new FileInputStream(inputfile);
		Reader decoder = new InputStreamReader(stream, "UTF-8");
		BufferedReader buffered = new BufferedReader(decoder);
		System.out.println("Loading file: " + inputfile + " to memory");
			
		String line=buffered.readLine();
		int count = 0;
		int failed = 0;
		int skip = 0;
		while(line != null)
        {
			JsonParser jsonParser = new JsonParser();
			JsonObject jsonObj = jsonParser.parse(line).getAsJsonObject();
			String topic = jsonObj.get("topic_id").getAsString();
			String status = jsonObj.get("status").getAsString();
			
			/*JsonElement tweet = jsonObj.get("json");
			if(tweet==null || tweet.isJsonNull())
			{
				skip++;
				line=buffered.readLine();
				continue;
			}*/
			
			String id = jsonObj.get("_id").toString();
			//String json = jsonObj.get("json").toString();
			//Document doc = Document.parse(json, false);
			
			count ++;
			
			StringBuffer sb = new StringBuffer();
			sb.append(id).append(",");
			sb.append(topic).append(",");
			sb.append(status);
			
			labeled.put(id, topic);
						
			if(count % 1000 == 0)
				System.out.println("Loaded " + count);
			
			line=buffered.readLine();
			
        }
		
		buffered.close();
		System.out.println("File loaded: "+count+" successful, "+ failed +" failed, "+ skip +" skipped.");
	}
}

