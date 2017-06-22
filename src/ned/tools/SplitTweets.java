package ned.tools;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Hashtable;
import java.util.Map.Entry;
import java.util.Set;

public class SplitTweets {
	private static Hashtable<String, ArrayList<String>> treesYes;
	private static Hashtable<String, ArrayList<String>> treesNo;

	public static void main(String[] args) throws IOException {
		treesYes = new Hashtable<String, ArrayList<String>> ();
		treesNo = new Hashtable<String, ArrayList<String>> ();
		
		String folder = "C:/data/Thesis/threads_petrovic_all/analysis_3m";
		String filename = "dataset_full_5m_V1.txt";
		
		String yesEvent = "events_yes.txt";
		String  noEvent = "events_no.txt";
		
		readTrees(folder+"/"+filename, yesEvent, noEvent);
		
		System.out.println(treesYes.size());
		System.out.println(treesNo.size());
		
		System.out.println("write to files...");
		PrintStream noOut = new PrintStream(new FileOutputStream(folder+"/"+noEvent));
		PrintStream yesOut = new PrintStream(new FileOutputStream(folder+"/"+yesEvent));
		
		yesOut.print("id,userId,created_at,timestamp,retweets,likes,parent,parentUserId,parentType,root,depth,time-lag,topic_id,is_topic\n");
		noOut.print("id,userId,created_at,timestamp,retweets,likes,parent,parentUserId,parentType,root,depth,time-lag,topic_id,is_topic\n");
		Collection<ArrayList<String>> values = treesYes.values();
		for (ArrayList<String> e : values)
		{
			for (String s : e)
			{
				yesOut.print(s);
				yesOut.print("\n");
			}
		}

		yesOut.close();
		
		
		Set<Entry<String, ArrayList<String>>> keyset = treesNo.entrySet();
		for (Entry<String, ArrayList<String>> e : keyset)
		{
			for (String s : e.getValue())
			{
				noOut.print(s);
				noOut.print("\n");
			}
		}
		noOut.close();
	}
	
	public static void readTrees(String dataset, String yesEvent, String noEvent) throws IOException 
	{	
		int parentDontExit = 0;
		FileInputStream stream = new FileInputStream(dataset);
		Reader decoder = new InputStreamReader(stream, "UTF-8");
		BufferedReader buffered = new BufferedReader(decoder);
		System.out.println("Processing file: " + dataset);
		
		
		String line = buffered.readLine();
		line = buffered.readLine();
		int counter = 0;
		while (line != null)
		{
			if (line.trim().isEmpty())
			{		
				line = buffered.readLine();
				continue;
			}
			
			String[] values = line.split(",");
			ArrayList<String> aa;
			if(values[9].equals(values[0]))
			{
				aa = new ArrayList<String>();
				aa.add(line);
				if(values[13].equals("yes"))
					treesYes.put(values[0], aa);
				else
					treesNo.put(values[0], aa);
			}
			else
			{
				if(values[13].equals("yes"))
				{
					aa = treesYes.get(values[9]);
					if(aa == null)
					{
						parentDontExit++;
						aa = new ArrayList<String>();
						aa.add(line);
						treesYes.put(values[9], aa);
					}
					
					treesYes.put(values[0], aa);
				} 
				else
				{
					aa = treesNo.get(values[9]);
					if(aa == null)
					{
						parentDontExit++;
						aa = new ArrayList<String>();
						aa.add(line);
						treesNo.put(values[9], aa);
					}
					treesNo.put(values[0], aa);
				}
				
			}
			
			line = buffered.readLine();
			counter++;
			
			if(counter % 100000 == 0)
				System.out.println(counter + ", " + parentDontExit);
			
			
		}
	
		
	}

}
