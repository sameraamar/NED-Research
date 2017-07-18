package com.javacodegeeks.examples;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintStream;
import java.io.Reader;
import java.io.UnsupportedEncodingException;
import java.io.Writer;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;

public class CleanCSVFile {

	private static BufferedReader reader;
	private static PrintStream writer;

	public static void main(String[] args) throws IOException {
		String DELIMITER = " ||| ";
		boolean firstIsHeader = true;
		String inputFilename = "C:/data/Thesis/threads_petrovic_all/full_all.txt.gz";
		String outputFilename = "C:\\Users\\t-saaama\\Documents\\Neo4j\\default.graphdb\\import\\full_all_clean.csv";
		
		openStreams(inputFilename, outputFilename);
		
		
		String line ;
		int count = 0;
		//String insertStatement = "INSERT INTO [thesis].[dbo].[short] ([leadId],[entropy],[users],[size],[text]) " +
		//						 "VALUES (" +
		//						 "<0>,<entropy, varchar(50),>,<users, varchar(50),>,<size, varchar(50),>,<text, varchar(255),>)";
		int headerLength = 0;
		while((line = reader.readLine())!=null)
		{
			line = line.replace(',', ' ');
			
			String[] values = line.split( Pattern.quote(DELIMITER) ) ;
			
			count++;
			
			if(count == 1)
			{
				//handle header...
				headerLength = values.length;
				//continue;
			}
	
			for(int k=0; k<values.length; k++)
			{
				if (k>=headerLength)
				{
					values[headerLength-1] += ' ' + values[k];
				}
				else if(count == 1)
					values[k] = cleanHeader(values[k]);
				else
					values[k] = cleanText(values[k]);
			}
			
			StringBuilder newLine = new StringBuilder();
			for(int k=0; k<headerLength; k++)
			{
				newLine.append(values[k]);
				if(k < headerLength-1)
					newLine.append(',');
			}
			newLine.append("\r\n");
			
			if(count % 10000 == 0)
				System.out.println(count);
			
			//if(count > 500000)
			//	break;
		
			writer.print(newLine.toString());
			
		}
		System.out.println(count);
		writer.close();
		reader.close();
	}

	private static String cleanText(String string) {
		return string.replace('\'', ' ').replace('"', ' ').replace(',', ' ');
	}

	private static String cleanHeader(String string) 
	{
		return string.replace('#', 'X');
	}

	private static void openStreams(String inputFilename, String outputFilename) throws IOException
	{
		InputStream input;
		if(inputFilename.endsWith(".gz"))
			input = new GZIPInputStream(new FileInputStream(inputFilename));
		else
			input = new FileInputStream(inputFilename);

		
		Reader decoder = new InputStreamReader(input, "UTF-8");
		reader = new BufferedReader(decoder);
		
		
		writer = new PrintStream(outputFilename);
	}
}
