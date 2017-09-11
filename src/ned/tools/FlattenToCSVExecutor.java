package ned.tools;

import java.io.PrintStream;
import java.util.Hashtable;
import java.util.concurrent.atomic.AtomicInteger;
import ned.types.Entry;
import ned.types.RedisBasedMap;

//import ned.types.HashtableRedis;

public class FlattenToCSVExecutor extends ProcessorExecutor {
	private PrintStream outFull;
	private PrintStream outShort;
	private RedisBasedMap<String, Entry> id2group;
	private Hashtable<String, String> positive;
	public AtomicInteger counter ;
	
	public FlattenToCSVExecutor(PrintStream fullOut, PrintStream shortOut, RedisBasedMap<String, Entry> id2group, Hashtable<String, String> positive, int number_of_threads)
	{
		super(number_of_threads);
		this.outFull = fullOut;
		this.outShort = shortOut;
		this.counter = new AtomicInteger(0);
		this.id2group = id2group;
		this.positive = positive;
		outFull.print( "id,userId,created_at,timestamp,retweets,likes,rtwt_likes,parent,parentUserId,parentType,root,depth,time-lag,text\n" );
		outShort.print( "id,root,topic_id,is_topic\n" );
	}
	
	protected Runnable createWorker(String line) {
		return new FlattenToCSVWorker(outFull, outShort, id2group, positive, counter, line);
	}

}
