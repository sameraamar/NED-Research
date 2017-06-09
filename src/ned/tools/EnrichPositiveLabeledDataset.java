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

	public static void main(String[] args) throws Exception {
		//getDocuments();
		String folder = "C:\\temp\\threads_petrovic_all\\mt_results";
		String filename = "C:\\temp\\threads_petrovic_all\\full_all.txt";

		Hashtable<String, String> labeled = new Hashtable<String, String>();
	
		//step1_1_loadLabeled("C:\\data\\events_db\\petrovic\\relevance_judgments_00000000", labeled);
		step1_2_LoadLabeled("C:\\temp\\threads_petrovic_all\\mechanical_turk_positive_labeled.txt", labeled);
		System.out.println( labeled );

		String filenameOut = folder+"\\has_topics.txt";
		step2_countHits(labeled, filename, filenameOut);
		

		String labeledFileName = folder+"\\positive_labeled.txt";
		step3_createLabeledDataset(filenameOut, labeledFileName);
		
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
		        
		        String[] values = line.split(Pattern.quote(DELIMITER));
		        
		        String topic = values[0];
		        String id = values[2];
		        String leadId = values[1];
		        String text = values[12] + values[13];
		        
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
	
	

	static public void step2_countHits(Hashtable<String, String> labeled, String filename, String filenameOut)
	{
		Map<String, String> cluster2Topic = markClustersWithTopics(filename, labeled);
		
		BufferedReader br = null;
		try {
			PrintStream out = new PrintStream(new FileOutputStream(filenameOut));

			br = new BufferedReader(new FileReader(filename));
		    String line = br.readLine();
		    //leadId\tid\tcreated\ttimestamp\tnearest\tdistance\tentropy\t#users\tsize\tage\tscore\ttext |||
		    
		    out.print(line + "\n");
			
		    int count = 0;
		    String leadId, topic;
	        int clustersCount = 0;
			while (line != null) {
		        line = br.readLine();
		        if(line==null || line.trim().equals(""))
		        	continue;
	        
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

		    	
		    	topic = cluster2Topic.get(leadId);
		    	
		    	if(topic != null)
		    	{
		    		out.print(topic + DELIMITER + line + "\n");
		    		clustersCount++;
		    	}
		    	
		        count++;
		        if(count % 500_000 == 0)
		        	System.out.println("processed: " + count);
		        
		    }
		    
		    
		    br.close();
		    out.close();
		    System.out.println("count lines: " + count + " , lines related to a topic: " + clustersCount );
		    
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
		    
		}
		
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
			String[] values = line.split("\t");
			String id = values[0];
			String topic = values[1];
			
			count ++;
			
			if(!labeled.contains(id))
				labeled.put(id, topic);
						
			if(count % 1000 == 0)
				System.out.println("Loaded " + count);
			
			line=buffered.readLine();
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

