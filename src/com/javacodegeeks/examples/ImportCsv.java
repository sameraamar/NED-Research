package com.javacodegeeks.examples;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.Reader;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.regex.Pattern;

import com.opencsv.CSVReader;

public class ImportCsv
{

		private static String inputFilename;
		private static BufferedReader reader;

		public static void main(String[] args)
		{
			String DELIMITER = " ||| ";
			boolean firstIsHeader = true;
			inputFilename = "C:/data/Thesis/threads_petrovic_all/short_all.txt";
			
			try {
				openStreams();
				readCsv(DELIMITER);
			
				readCsvUsingLoad();
			} catch (Exception e)
			{
			
				
			}
			
			
			try {
				reader.close();
			} catch (IOException e) {
				e.printStackTrace();
			}

		}

		private static void openStreams() throws IOException
		{
			FileInputStream input = new FileInputStream(inputFilename);
			
			Reader decoder = new InputStreamReader(input, "UTF-8");
			reader = new BufferedReader(decoder);
		}

		private static void readCsv(String sep)
		{

				try (Connection connection = DBConnection.getConnection();)
				{
						String insertQuery = "Insert into txn_tbl (txn_id,txn_amount, card_number, terminal_id) values (null,?,?,?)";
						PreparedStatement pstmt = connection.prepareStatement(insertQuery);
						String line = null;
						int i = 0;
						int fieldCount = 1;
						int batchExecute = 10;
						ArrayList<String> fielNames = null;
						while((line = reader.readLine()) != null)
						{
							String[] rawData = line.split( Pattern.quote(sep) ) ;

							if(i == 0)
							{
								//handle header
								fieldCount = rawData.length;
								batchExecute = fieldCount * 10;
								fielNames = new ArrayList<String>();
								for(int j = 0; j<fieldCount; j++)
									fielNames.add(rawData[j]);
								
								continue;
							}
							pstmt.setString((i % 3) + 1, line);

							if (++i % fieldCount == 0)
									pstmt.addBatch();// add batch

							if (i % batchExecute == 0)// insert when the batch size is 10
									pstmt.executeBatch();
						}
						System.out.println("Data Successfully Uploaded");
				}
				catch (Exception e)
				{
						e.printStackTrace();
				}

		}

		private static void readCsvUsingLoad()
		{
				try (Connection connection = DBConnection.getConnection())
				{

						String loadQuery = "LOAD DATA LOCAL INFILE '" + "C:\\upload.csv" + "' INTO TABLE txn_tbl FIELDS TERMINATED BY ','"
										+ " LINES TERMINATED BY '\n' (txn_amount, card_number, terminal_id) ";
						System.out.println(loadQuery);
						Statement stmt = connection.createStatement();
						stmt.execute(loadQuery);
				}
				catch (Exception e)
				{
						e.printStackTrace();
				}
		}

		
}
