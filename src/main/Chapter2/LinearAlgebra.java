/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package main.Chapter2;

import java.util.Arrays;
import java.util.List;
import org.apache.mxnet.javaapi.Context;
import org.apache.mxnet.javaapi.DType;
import org.apache.mxnet.javaapi.NDArray;
import org.apache.mxnet.javaapi.Shape;
import org.apache.mxnet.javaapi.meanParam;
import org.apache.mxnet.javaapi.sumParam;
import org.apache.mxnet.javaapi.sum_axisParam;

/**
 *
 * @author omar
 */
public class LinearAlgebra {

    public static void main(String[] args) {
        //Scalaras 
        Context context = Context.cpu();
        NDArray x = new NDArray(new float[]{3}, new Shape(new int[]{1}), context);
        NDArray y = new NDArray(new float[]{2}, new Shape(new int[]{1}), context);

        System.out.println("X+Y: " + Arrays.toString(x.add(y).toArray()));
        System.out.println("X*Y: " + Arrays.toString(x.multiply(y).toArray()));
        System.out.println("X/Y: " + Arrays.toString(x.div(y).toArray()));
        System.out.println("X**Y: " + Arrays.toString(x.pow(y).toArray()));

        //Vectors 
        NDArray vector = NDArray.arange(0, 3, 1, 1, context, DType.Float64());
        System.out.println("Vectors:" + Arrays.toString(vector.toArray()));
        System.out.println("Vector 2nd Element" + Arrays.toString(vector.slice(2).toArray()));
        System.out.println("Vextor Length:" + vector.size());
        System.out.println("Vector Shape:" + Arrays.toString(vector.shape().toArray()));

        //Matrices
        NDArray matrixThreeByTwo = NDArray.arange(0, 6, 1, 1, context, DType.Int32());
        matrixThreeByTwo = matrixThreeByTwo.reshape(new int[]{3, 2});

        System.out.println("Matrix:" + matrixThreeByTwo.toString());
        System.out.println("Matrix:" + matrixThreeByTwo.T().toString());
        //Matrix are  equal to their T // A==A.T if they symmetric 
        NDArray symmetricMatrix = new NDArray(new float[]{1, 2, 3, 2, 0, 4, 3, 4, 5}, new Shape(new int[]{3, 3}), context);
        System.out.println("A==A.T" + NDArray.equal(symmetricMatrix, symmetricMatrix.T()).toString());

        //Tnsors 
        NDArray tensor = NDArray.arange(0, 24, 1, 1, context, DType.Float64());
        tensor = tensor.reshape(new int[]{2, 3, 4});
        System.out.println("Matrix:" + tensor.toString());

        System.out.println("===");
        //Basic Properties of Tensor Arithmetic
        NDArray a = NDArray.arange(0, 6, 1, 1, context, DType.Float64());
        a = a.reshape(new int[]{2, 3});
        System.out.println("Array A:" + a.toString());
        NDArray b = null;
        b = a.copy();
        System.out.println("Array B:" + b.toString());
        System.out.println("X+Y: " + a.add(b).toString());
        System.out.println("X+Y: " + a.multiply(b).toString());
        System.out.println("===");
        //Tensor * scalar ;
        System.out.println("2+Tensor: " + tensor.add(2).toString());
        System.out.println("2*Tensor: " + tensor.multiply(2).shape().toString());
        System.out.println("===");
        
        //Reduction

        sumParam p = new sumParam(vector);
        System.out.println("Vectors:" + Arrays.toString(vector.toArray()));
        System.out.println("vertor Sum: " + Arrays.toString(NDArray.sum(p)[0].toArray()));
        System.out.println("===");
        p = new sumParam(a);
        System.out.println("Matrix Shape:" + Arrays.toString(a.shape().toArray()));
        System.out.println("Matrix Sum: " + Arrays.toString(NDArray.sum(p)[0].toArray()));
        sum_axisParam p1 = new sum_axisParam(a);
        System.out.println("===");
        p1.setAxis(new Shape(List.of(0)));
        //System.out.println("Shape " + Arrays.toString(p1.getAxis().shape().toArray()));
        System.out.println("Matrix Sum axis zero Shape: " + Arrays.toString(NDArray.sum_axis(p1)[0].shape().toArray()));
        p1.setAxis(new Shape(List.of(1)));
        System.out.println("Matrix Sum axis one Shape: " + Arrays.toString(NDArray.sum_axis(p1)[0].shape().toArray()));
        p1.setAxis(new Shape(List.of(0, 1)));
        System.out.println("Matrix Sum axis one and Zero == array.Sum : " + NDArray.equal(NDArray.sum_axis(p1)[0], a.sum(p)[0]).toString());
        System.out.println("===");
        meanParam meanParam = new meanParam(a);
        //Matrix Mean == Matrix Sum/Size
        System.out.println("Matrix Mean" + NDArray.mean(meanParam)[0].toString());
        System.out.println("Matrixc Sum/size" + NDArray.sum(p)[0].div(a.size()).toString());
        //Matrix Mean == Matrix Sum/Size // Using 0 Axis
        meanParam.setAxis(new Shape(List.of(0)));
        p1.setAxis(new Shape(List.of(0)));
        System.out.println("===");
        System.out.println("Matrix Mean axis 0:" + NDArray.mean(meanParam)[0].toString());
        System.out.println("Matrixc Sum/size axis 0:" + NDArray.sum_axis(p1)[0].div(a.shape().toArray()[0]).toString());
        System.out.println("===");
        //Non-Reduction Sum
         
        

    }

}
