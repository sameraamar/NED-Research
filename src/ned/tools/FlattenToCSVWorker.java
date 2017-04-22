package ned.tools;

import java.io.PrintStream;
import ned.types.Document;
import ned.types.GlobalData;

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
		sb.append("i").append(id).append(",");
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
			parent = "i" + jRply; 
			parentUser = doc.getReplyToUserId();
			parentType = "1";
		}
		if(jRtwt != null && !jRtwt.isEmpty())
		{
			parent = "i" + jRtwt ; 
			parentUser = doc.getRetweetedUserId();
			parentType = "2";
		}
		
		sb.append(parent).append(",");
		sb.append(parentUser).append(",");
		sb.append(parentType).append(",");
		
		
		Entry entry = FlattenDatasetMain_Step1.id2group.get( id );
		String root = "";
		int level = 0;
		long timeLag = -1;
		if(entry != null)
		{
			root = "i" + entry.leadId;
			level = entry.level;
			//Document rootDoc = GlobalData.getInstance().getDocumentFromRedis("id2doc_parser", entry.leadId);
			timeLag = -1;
			timeLag = timestamp - entry.firstTimestamp;
		}

		sb.append(root).append(",");
		sb.append(level).append(",");
		sb.append(timeLag).append(",");
		Integer topic_id = FlattenDatasetMain_Step1.getTopic( id );
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