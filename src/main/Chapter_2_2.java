package main;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 *
 * @author omar
 */
public class Chapter_2_2 {

    /**
     *
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        File file = new File("../data/");
        file.mkdir();

        String dataFile = "../data/house_tiny.csv";

// Create file
        File f = new File(dataFile);
        f.createNewFile();

// Write to file
        try (FileWriter writer = new FileWriter(dataFile)) {
            writer.write("NumRooms,Alley,Price\n"); // Column names
            writer.write("NA,Pave,127500\n");  // Each row represents a data example
            writer.write("2,NA,106000\n");
            writer.write("4,NA,178100\n");
            writer.write("NA,NA,140000\n");
        }
        catch(Exception e){
            
        }
    }

}
