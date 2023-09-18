package main.Chapter2.mxnet;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import org.apache.mxnet.javaapi.Context;
import org.apache.mxnet.javaapi.NDArray;
import org.apache.mxnet.javaapi.Shape;
import tech.tablesaw.api.BooleanColumn;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.IntColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.columns.Column;

/**
 *
 * @author omar
 */
public class DataPreprocessing {

    /**
     *
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {

        Context context = Context.cpu();
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
        } catch (Exception e) {
            System.out.println("Error file wirter" + e.getMessage());
        }
        System.out.println("f:" + f.getCanonicalPath());
        Table data = Table.read().csv(f.getCanonicalPath());

        System.out.println(data.structure().printAll());

        System.out.println(data.printAll());

        Table input = data.create(data.columns());
        input.removeColumns("Price");

        Table output = data.selectColumns("Price");

        Column col = input.column("NumRooms");
        col.set(col.isMissing(), (int) input.nCol("NumRooms").mean());

        System.out.println("Output");
        System.out.println(output.printAll());
        System.out.println("Input");
        System.out.println(input.printAll());

        StringColumn colAlley = (StringColumn) input.column("Alley");
        List<BooleanColumn> dummies = colAlley.getDummies();
        input.removeColumns(colAlley);
        input.addColumns(DoubleColumn.create("Alley_Pave", dummies.get(0).asDoubleArray()),
                DoubleColumn.create("Alley_nan", dummies.get(1).asDoubleArray())
        );

        System.out.println("Input After creating dummies from Alley Column");
        System.out.println(input.printAll());

        double[][] InputMatrix = input.as().doubleMatrix();
        double[][] outputMatrix = output.as().doubleMatrix();
        //To collections or to arrays 
        double[] InputArray = Arrays.stream(InputMatrix).flatMapToDouble(Arrays::stream).toArray();
        double[] outputArray = Arrays.stream(outputMatrix).flatMapToDouble(Arrays::stream).toArray();

        Shape inputShape = new Shape(new int[]{InputMatrix.length, InputMatrix[InputMatrix.length - 1].length});
        Shape outputShape = new Shape(new int[]{outputMatrix.length, outputMatrix[outputMatrix.length - 1].length});
        NDArray inputNDArray = new NDArray(InputArray, inputShape, context);
        NDArray outputNDArray = new NDArray(outputArray, outputShape, context);

        System.out.println("Input NDArray" + inputNDArray.toString());
        System.out.println("Output NDArray" + outputNDArray.toString());

        
        //Trying to add new columns based on number of rooms.  
        IntColumn colRooms = (IntColumn) input.column("NumRooms");

        input.addColumns(DoubleColumn.create("Stdiuo", colRooms.asList().stream().map(e -> e <= 2 ? 1 : 0).toList()));
        input.addColumns(DoubleColumn.create("Wing", colRooms.asList().stream().map(e -> e > 2 ? 1 : 0).toList()));

       System.out.println(input.printAll());

    }
}
