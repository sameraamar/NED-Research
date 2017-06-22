package ned.graphstream;
import java.awt.Color;
import java.util.List;
import java.util.Random;

import org.graphstream.algorithm.ConnectedComponents;
import org.graphstream.graph.Graph;
import org.graphstream.graph.Node;
import org.graphstream.graph.implementations.SingleGraph;

public class TestGraphStream {

	public static void main(String[] args) {
		 
		Graph graph = new SingleGraph("Tutorial 1");
		//graph.display();

		Random rnd = new Random();
		for (int i=0; i<20_000_000; i++)
		{
			Node n = graph.addNode("A"+i );
			//if(rnd.nextInt() % 10 == 0)
			//	n.addAttribute("ui.class", "important");
			
			if (i<6)
				continue;
			
			int neighbors = rnd.nextInt() % 5;
			for (int j=0; j<neighbors; j++)
			{
				int v = rnd.nextInt(i-5)+4;
				int direction = 0; //rnd.nextInt()%2;
				String trgt;
				String e;
				String src;
				if(direction==0)
				{
					e = i+","+v;
					src = "A"+i;
					trgt = "A"+v;
				}
				else
				{
					e = v+","+i;
					src = "A"+v;
					trgt = "A"+i;
				}
				if(graph.getEdge(e) == null)
					graph.addEdge(e, src, trgt);
			}
			
//			try {
//				Thread.sleep(500);
//			} catch (InterruptedException e) {
//				e.printStackTrace();
//			}
		}

		ConnectedComponents cc = new ConnectedComponents();
        cc.init(graph);
        
        List<Node> nods = cc.getGiantComponent();
        System.out.println("giant component: " + nods.size());
        for(Node n : nods)
        {
        	//n.addAttribute("ui.color", Color.RED);
        	n.addAttribute("ui.style", "fill-color: rgb(100,0,0);");
        	n.addAttribute("ui.style", "fill-color: red;");
        	//graph.addAttribute(arg0, arg1);
        }
        
        System.out.printf("%d connected component(s) in this graph, so far.%n",
                cc.getConnectedComponentsCount());
	}
}
