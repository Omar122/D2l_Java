/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package main.Chapter2;

/**
 *
 * @author omar
 *
 */
import java.io.IOException;
import static java.lang.System.exit;
import java.util.List;
import java.util.Arrays;
import org.apache.mxnet.javaapi.Context;
import org.apache.mxnet.javaapi.DType;
import org.apache.mxnet.javaapi.NDArray;
import org.apache.mxnet.javaapi.Shape;
import org.apache.mxnet.javaapi.random_normalParam;
import org.apache.mxnet.javaapi.sumParam;
import org.bytedeco.javacpp.Loader;
import static java.lang.System.out;
import static java.lang.System.setProperty;

public class DataManipulation {

    public static void main(String[] args) throws IOException {
        Loader.load(org.bytedeco.mxnet.global.mxnet.class);
        setProperty("org.bytedeco.openblas.load", "mkl");
        Context context = Context.cpu();

        out.println((char) 27 + "[32m" + "-Getting Started-" + (char) 27 + "[0m");
        //args:(start value , end value , step ,repeat , context, and type) 
        NDArray array = NDArray.arange(0, 12, 1, 1, context, DType.Int32());

        out.println("x: " + array.toString());
        out.println("x Sizr: " + array.size());
        out.println("x Shape: " + array.shape());

        array = array.reshape(new int[]{3, 4});
        out.println("x ReShaped to 3,4: " + array.toString());
        //array = NDArray.arange(0, 12, 1, 1, context, DType.Int32());
        out.println("x: " + array.toString());

        array = array.reshape(new int[]{-1, 4});
        out.println("x ReShaped to using -1: " + array.toString());
        out.println("---");

        random_normalParam randomParam = new random_normalParam();
        randomParam.setShape(new Shape(new int[]{3, 4}));
        randomParam.setScale(Float.valueOf(1));
        randomParam.setLoc(Float.valueOf(0));
        NDArray[] randomArray = NDArray.random_normal(randomParam);

        for (NDArray nDArray : randomArray) {
            out.println("Random Aray " + nDArray.toString());
        }

        NDArray construct_array = new NDArray(new float[]{2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1}, new Shape(new int[]{3, 4,}), context);
        out.println("constructed Aray " + construct_array.toString());

        out.println("---");
        NDArray ones = NDArray.ones(context, List.of(2, 3, 4));
        NDArray zeros = NDArray.zeros(context, List.of(2, 3, 4));
        out.println("Ones: " + Arrays.toString(ones.toArray()));
        out.println("---");
        out.println("Zeros: " + Arrays.toString(zeros.toArray()));

        out.println((char) 27 + "[32m" + "-Indexing and Slicing-" + (char) 27 + "[0m");
        out.println("Array: " + Arrays.toString(array.toArray()));

        NDArray array_1to3Slice = array.slice(1, 3);
        NDArray array_Neg1 = array.slice(1, 3);
        out.println("Array sliced 1_3" + Arrays.toString(array_1to3Slice.toArray()));
        out.println("Array sliced -1" + Arrays.toString(array_Neg1.toArray()));

        out.println("---");
        out.println("---");
        out.println((char) 27 + "[32m" + "Operations" + (char) 27 + "[0m");

        NDArray expArray = null;
        expArray = array.copy();

        NDArray.exp(array, expArray);
        out.println("Exp Array: " + Arrays.toString(expArray.toArray()));

        NDArray xArray = new NDArray(new float[]{1, 2, 4, 8}, new Shape(new int[]{4}), context);

        NDArray yArray = new NDArray(new float[]{2, 2, 2, 2}, new Shape(new int[]{4}), context);

        out.println("X Array: " + Arrays.toString(xArray.toArray()));
        out.println("Y Array: " + Arrays.toString(yArray.toArray()));

        out.println("X+Y: " + Arrays.toString(xArray.add(yArray).toArray()));
        out.println("X-Y: " + Arrays.toString(xArray.subtract(yArray).toArray()));
        out.println("X*Y: " + Arrays.toString(xArray.multiply(yArray).toArray()));
        out.println("X/Y: " + Arrays.toString(xArray.div(yArray).toArray()));
        out.println("X**Y: " + Arrays.toString(xArray.pow(yArray).toArray()));

        out.println("-concatenate-");

        NDArray x = NDArray.arange(0, 12, 1, 1, context, DType.Float32());
        x = x.reshape(new int[]{3, 4});

        NDArray y = new NDArray(new float[]{2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1}, new Shape(new int[]{3, 4,}), context);
        out.println("X: " + x.toString());
        out.println("Y: " + y.toString());

        //axis Zero is x axis One is the Y axis
        //arg(the list of arrays , numbe of array in the list axis and output) 
        NDArray reuslt = NDArray.empty(context, new int[]{6, 4});
        NDArray.concat(new NDArray[]{x, y}, 2, 0, reuslt);

        out.println("reuslt contact aixs 0: " + reuslt.toString());
        reuslt = NDArray.empty(context, new int[]{3, 8});
        NDArray.concat(new NDArray[]{x, y}, 2, 1, reuslt);
        out.println("reuslt contact aixs 1: " + reuslt.toString());

        out.println("X==y" + NDArray.equal(x, y).toString());

        sumParam p = new sumParam(x);

        out.println("X Sum: " + Arrays.toString(NDArray.sum(p)[0].toArray()));

        out.println((char) 27 + "[32m" + "-1.4. Broadcasting-" + (char) 27 + "[0m");

        NDArray a = NDArray.arange(0, 3, 1, 1, context, DType.Int32());
        a = a.reshape(new int[]{3, 1});
        out.println("a:" + a.toString());
        NDArray b = NDArray.arange(0, 2, 1, 1, context, DType.Int32());
        b = b.reshape(new int[]{1, 2});
        out.println("b:" + b.toString());
        out.println("a+b:" + NDArray.broadcast_add(a, b, null)[0].toString());

        ///1.5. Saving Memory 
        //I dont think it is meant for us here for us here
        out.println("---add and addinplace---");

        a = NDArray.arange(0, 3, 1, 1, context, DType.Int32());
        b = NDArray.arange(1, 4, 1, 1, context, DType.Int32());

        out.println("a:" + a.toString());
        //Save Memoery by using addinplace
        NDArray c = a.add(b);
        out.println("a != c" +Arrays.toString(NDArray.equal(a, c).toArray()));
        a = NDArray.arange(0, 3, 1, 1, context, DType.Int32());
        b = NDArray.arange(1, 4, 1, 1, context, DType.Int32());
        c = a.addInplace(b);
        out.println("a == c" +Arrays.toString(NDArray.equal(a, c).toArray()));
        
        
        exit(0);

    }

}
