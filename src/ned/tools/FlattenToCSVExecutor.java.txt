package ned.tools;

import java.io.PrintStream;
import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultEdge;

//import ned.types.HashtableRedis;

public class FlattenToCSVExecutor extends ProcessorExecutor {
	PrintStream out;
	private Graph<String, DefaultEdge> graph;
	
	//private HashtableRedis<String> id2group;
	
	public FlattenToCSVExecutor(PrintStream out, Graph<String, DefaultEdge> g, int number_of_threads)
	{
		super(number_of_threads);
		this.out = out;
		this.graph = g;
		out.print( "id,userId,created_at,timestamp,retweets,likes,parent,parentUserId,parentType,root,depth,time-lag,topic_id,is_topic,$\n" );
	}
	
	protected Runnable createWorker(String line) {
		return new FlattenToCSVWorker(out, graph, line);
	}

}
