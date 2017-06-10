package ned.types;

import java.io.Serializable;

public class Entry implements Serializable, DirtyBit {
	/**
	 * 
	 */
	private static final long serialVersionUID = 6575245165701295855L;
	private String leadId;
	private int level;
	private long firstTimestamp;
	transient boolean dirty;
	
	public Entry(String leadId, long firstTimestamp, int level)
	{
		this.leadId = leadId;
		this.firstTimestamp = firstTimestamp;
		this.level = level;
		dirtyOn();
	}

	@Override
	public boolean isDirty() {
		return dirty;
	}

	@Override
	public void dirtyOff() {
		dirty = false;
	}

	@Override
	public void dirtyOn() {
		dirty = true;
	}

	public String getLeadId() {
		return leadId;
	}

	//public void setLeadId(String leadId) {
	//	this.leadId = leadId;
	//}

	public int getLevel() {
		return level;
	}

	//public void setLevel(int level) {
	//	this.level = level;
	//}

	public long getFirstTimestamp() {
		return firstTimestamp;
	}

	//public void setFirstTimestamp(long firstTimestamp) {
	//	this.firstTimestamp = firstTimestamp;
	//}
}
