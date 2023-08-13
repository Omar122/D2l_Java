/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package main;

/**
 *
 * @author omar
 *
 */
import java.io.IOException;
import java.util.List;
import org.apache.mxnet.javaapi.Context;
import org.apache.mxnet.javaapi.DType;
import org.apache.mxnet.javaapi.NDArray;
import org.apache.mxnet.javaapi.Shape;
import org.apache.mxnet.javaapi.random_normalParam;
import org.bytedeco.javacpp.Loader;

public class Chapter2 {

    public static void main(String[] args) throws IOException {
        Loader.load(org.bytedeco.mxnet.global.mxnet.class);
        System.setProperty("org.bytedeco.openblas.load", "mkl");
        Context context = Context.cpu();

        System.out.println((char) 27 + "[32m" + "-Getting Started-" + (char) 27 + "[0m");

        NDArray array = NDArray.arange(0, 12, 1, 1, context, DType.Int32());

        System.out.println("x: " + array.toString());
        System.out.println("x Sizr: " + array.size());
        System.out.println("x Shape: " + array.shape());
        int[] x = {3, 4};
        array = array.reshape(x);
        System.out.println("x ReShaped to 3,4: " + array.toString());
        //array = NDArray.arange(0, 12, 1, 1, context, DType.Int32());
        System.out.println("x: " + array.toString());
        int[] y = {-1, 4};
        array = array.reshape(y);
        System.out.println("x ReShaped to using -1: " + array.toString());
        System.out.println("---");

        random_normalParam randomParam = new random_normalParam();
        randomParam.setShape(new Shape(x));
        randomParam.setScale(Float.valueOf(1));
        randomParam.setLoc(Float.valueOf(0));
        NDArray[] randomArray = NDArray.random_normal(randomParam);

        for (NDArray nDArray : randomArray) {
            System.out.println("Random Aray " + nDArray.toString());
        }

        NDArray construct_array = new NDArray(new float[]{2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1}, new Shape(new int[]{3, 4,}), context);
        System.out.println("constructed Aray " + construct_array.toString());

        System.out.println("---");
        NDArray ones = NDArray.ones(context, List.of(2, 3, 4));
        NDArray zeros = NDArray.zeros(context, List.of(2, 3, 4));
        System.out.println("Ones: " + ones.toString());
        System.out.println("---");
        System.out.println("Zeros: " + zeros.toString());

        System.out.println((char) 27 + "[32m" + "-Indexing and Slicing-" + (char) 27 + "[0m");
        System.out.println("Array: " + array.toString());

        NDArray array_1to3Slice = array.slice(1, 3);
        NDArray array_Neg1 = array.slice(1, 3);
        System.out.println("Array sliced 1_3" + array_1to3Slice.toString());
        System.out.println("Array sliced -1" + array_Neg1.toString());

        System.out.println("---");
        System.out.println("---");
        System.out.println((char) 27 + "[32m" + "Operations" + (char) 27 + "[0m");

        NDArray expArray = null;
        expArray = array.copy();
        System.err.println("Exp Array" + expArray.toString());
        NDArray.exp(array, expArray);
        System.err.println("Exp Array" + expArray.toString());

        NDArray xArray = new NDArray(new float[]{1, 2, 4, 8}, new Shape(new int[]{4}), context);

        NDArray yArray = new NDArray(new float[]{2, 2, 2, 2}, new Shape(new int[]{4}), context);

        System.err.println("X Array" + xArray.toString());
        System.err.println("Y Array" + yArray.toString());
        
        
        System.exit(0);

    }

}
