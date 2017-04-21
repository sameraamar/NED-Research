package ned.tools;

public class Entry {
	String leadId;
	int level;
	long firstTimestamp;
	
	public Entry(String leadId, long firstTimestamp, int level)
	{
		this.leadId = leadId;
		this.firstTimestamp = firstTimestamp;
		this.level = level;
	}
}
