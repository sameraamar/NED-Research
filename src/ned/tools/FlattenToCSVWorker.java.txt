package ned.tools;

import java.io.PrintStream;
import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultEdge;
import ned.types.Document;
import ned.types.Entry;

public class FlattenToCSVWorker extends ProcessorWorker
{
	private static final String PREFIX = "";
	PrintStream out;
	private Graph<String, DefaultEdge> graph;

	
    public FlattenToCSVWorker(PrintStream out, Graph<String, DefaultEdge> graph, String docJson)
    {
        super( docJson );
        this.out = out;
        this.graph = graph;
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
			return;
		
		
		writeToCSV(doc);
	
		
		
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
		
		if(graph != null)
		{
			synchronized (graph)
			{
				if (!parent.trim().equals("") && !graph.containsVertex(parent))
					graph.addVertex(parent);
				
				if (!graph.containsVertex(tmpId))
					graph.addVertex(tmpId);
			}
		}
		
		//CustomEdge e = new CustomEdge(parentType);
		//if(!parent.trim().equals("") && graph.containsVertex(parent)) 
		//	graph.addEdge(tmpId,  parent);
		
		//System.out.println( graph.edgeSet().size() );
		
		sb.append(parent).append(",");
		sb.append(parentUser).append(",");
		sb.append(parentType).append(",");
		
		
		Entry entry = FlattenDatasetMain_Step1.id2group2.get( id );
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
		Integer topic_id = -1; //FlattenDatasetMain_Step1.getTopic( id );
		sb.append(topic_id).append(",");
		sb.append(topic_id == -1 ? "no" : "yes").append(",");
		sb.append("$\n");
		
		out.print(sb.toString());
    }

	private String handleNull(String value) 
	{
		return value==null ? "" : value; 
	}

}