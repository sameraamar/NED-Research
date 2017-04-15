package ned.tools;

import java.io.PrintStream;
import ned.types.Document;

public class FlattenToCSVWorker extends ProcessorWorker
{
	PrintStream out;
	
    public FlattenToCSVWorker(PrintStream out, String docJson)
    {
        super( docJson );
        this.out = out;
    }

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
		int retweets = doc.getRetweetCount();
		String created_at = doc.getCreatedAt();
		String id = doc.getId();
		long timestamp = doc.getTimestamp();
		String userId = doc.getUserId();
		int likes = doc.getFavouritesCount();
		
		StringBuffer sb = new StringBuffer();
		sb.append(id).append(",");
		sb.append(userId).append(",");
		sb.append(created_at).append(",");
		sb.append(timestamp).append(",");
		sb.append(retweets).append(",");
		sb.append(likes).append(",");
		
		String parent = "";
		String parentType = "";
		if(jRply!=null && !jRply.isEmpty())
		{
			parent = jRply; 
			parentType = "1";
		}
		if(jRtwt != null && !jRtwt.isEmpty())
		{
			parent = jRtwt ; 
			parentType = "2";
		}
		
		sb.append(parent).append(",");
		sb.append(parentType).append(",");
		
		
		Entry entry = FlattenDatasetMain.id2group.get( id );
		String root = "";
		int level = 0;
		if(entry != null)
		{
			root = entry.leadId;
			level = entry.level;
		}

		sb.append(root).append(",");
		sb.append(level).append(",");
		Integer topic_id = FlattenDatasetMain.getTopic( id );
		sb.append(topic_id).append(",");
		sb.append(topic_id > -1 ? "yes" : "no").append(",");
		sb.append("$");
		
		out.println(sb.toString());
    }

	private String handleNull(String value) 
	{
		return value==null ? "" : value; 
	}

}