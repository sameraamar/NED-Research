package ned.tools;

import java.io.PrintStream;
import java.util.Hashtable;
import java.util.concurrent.atomic.AtomicInteger;

import ned.types.Document;
import ned.types.Entry;
import ned.types.RedisBasedMap;

public class FlattenToCSVWorker extends ProcessorWorker
{
	private static final String PREFIX = "";
	PrintStream outFull;
	PrintStream outShort;
	RedisBasedMap<String, Entry> id2group;
	Hashtable<String, String> positive;
	AtomicInteger counter;

	
    public FlattenToCSVWorker(PrintStream outFull, PrintStream outShort, RedisBasedMap<String, Entry> id2group, Hashtable<String, String> positive, AtomicInteger counter, String docJson)
    {
        super( docJson );
        this.outFull = outFull;
        this.outShort = outShort;
        this.id2group = id2group;
        this.positive = positive;
        this.counter = counter;
    }

    
    /*

    try {
        URL amazon = new URL("http://www.amazon.com");
        URL yahoo = new URL("http://www.yahoo.com");
        URL ebay = new URL("http://www.ebay.com");

        // add the vertices
        g.addVertex(amazon);
        g.addVertex(yahoo);
        g.addVertex(ebay);

        // add edges to create linking structure
        g.addEdge(yahoo, amazon);
        g.addEdge(yahoo, ebay);
    } catch (MalformedURLException e) {
        e.printStackTrace();
    }

	*/
    
    @Override
    protected void processCommand() 
    {

    	Document doc = Document.parse(docJson, false);
		if(doc == null)
		{
			this.counter.incrementAndGet();
			return;
		}
		
		writeToCSV(doc);
	
		this.counter.incrementAndGet();
		
    }
    
    protected void writeToCSV(Document doc) 
    {
		String jRply = handleNull(doc.getReplyTo());
		
		String jRtwt = handleNull(doc.getRetweetedId());
		
		String jQuote = handleNull(doc.getQuotedStatusId());
		
		int retweets = doc.getRetweetCount();
		String created_at = doc.getCreatedAt();
		String id = doc.getId();
		long timestamp = doc.getTimestamp();
		String userId = doc.getUserId();
		int likes = doc.getFavouritesCount();
		
		StringBuffer sb = new StringBuffer();
		String tmpId = PREFIX + id;
		sb.append(tmpId).append(",");
		sb.append(userId).append(",");
		sb.append(created_at).append(",");
		sb.append(timestamp).append(",");
		sb.append(retweets).append(",");
		sb.append(likes).append(",");
		
		String parent = "";
		String parentType = "";
		String parentUser = "";
		if(jRply!=null && !jRply.isEmpty())
		{
			parent = PREFIX + jRply; 
			parentUser = doc.getReplyToUserId();
			parentType = "1";
		}

		if(jRtwt != null && !jRtwt.isEmpty())
		{
			parent = PREFIX + jRtwt ; 
			parentUser = doc.getRetweetedUserId();
			parentType = "2";
		}
		
		if(jQuote != null && !jQuote.isEmpty())
		{
			parent = PREFIX + jQuote ; 
			parentUser = doc.getQuotedUserId();
			parentType = "3";
		}
		
		sb.append(parent).append(",");
		sb.append(parentUser).append(",");
		sb.append(parentType).append(",");
		
		
		Entry entry = id2group.get( id );
		String root = "";
		int level = 0;
		long timeLag = -1;
		if(entry != null)
		{
			root = PREFIX + entry.getLeadId();
			level = entry.getLevel();
			//Document rootDoc = GlobalData.getInstance().getDocumentFromRedis("id2doc_parser", entry.leadId);
			timeLag = -1;
			timeLag = timestamp - entry.getFirstTimestamp();
		}

		sb.append(root).append(",");
		sb.append(level).append(",");
		sb.append(timeLag).append(",");
		String topicId = positive.get( id );
		if(topicId == null || topicId.equals("-1"))
			topicId = "";
		sb.append(topicId).append(",");
		sb.append(topicId.equals("") ? "no" : "yes");
		sb.append("\n");
		
		outFull.print(sb.toString());
		
		
		sb = new StringBuffer();
		sb.append(tmpId).append(",");
		sb.append(root).append(",");
		sb.append(topicId).append(",");
		sb.append(topicId.equals("") ? "no" : "yes");
		sb.append("\n");
		
		outShort.print(sb.toString());
    }

	private String handleNull(String value) 
	{
		return value==null ? "" : value; 
	}

}