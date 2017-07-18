package ned.types;

import java.io.Serializable;

public class Entry implements Serializable, DirtyBit {
	/**
	 * 
	 */
	private static final long serialVersionUID = 6575245165701295855L;
	private String leadId;
	private String parentType;
	private int level;
	private long timestamp;
	private long time_delta;
	transient boolean dirty;
	
	public Entry(String parentId, String parentType, long timestamp, int level)
	{
		this.leadId = parentId;
		this.parentType = parentType;
		this.timestamp = timestamp;
		this.time_delta = 0;
		this.level = level;
		dirtyOn();
	}

	public Entry(String parentId, String parentType, long firstTimestamp, long thisTimeStamp, int level) {
		this.leadId = parentId;
		this.parentType = parentType;
		this.timestamp = thisTimeStamp;
		this.time_delta = thisTimeStamp-firstTimestamp;
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

	public String getParentId() {
		return leadId;
	}

	public long getTimeDelta()
	{
		return time_delta;
	}


	public int getLevel() {
		return level;
	}

	public long getTimestamp() {
		return timestamp;
	}

	public String getParentType() {
		return parentType;
	}

}
