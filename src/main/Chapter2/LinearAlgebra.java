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
import org.apache.mxnet.javaapi.cumsumParam;
import org.apache.mxnet.javaapi.dotParam;
import org.apache.mxnet.javaapi.meanParam;
import org.apache.mxnet.javaapi.sumParam;
import org.apache.mxnet.javaapi.sum_axisParam;
import static java.lang.System.*;
import org.apache.mxnet.javaapi.normParam;

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
        
        out.println("X+Y: " + Arrays.toString(x.add(y).toArray()));
        out.println("X*Y: " + Arrays.toString(x.multiply(y).toArray()));
        out.println("X/Y: " + Arrays.toString(x.div(y).toArray()));
        out.println("X**Y: " + Arrays.toString(x.pow(y).toArray()));

        //Vectors 
        NDArray vector = NDArray.arange(0, 3, 1, 1, context, DType.Float64());
        out.println("Vectors:" + Arrays.toString(vector.toArray()));
        out.println("Vector 2nd Element" + Arrays.toString(vector.slice(2).toArray()));
        out.println("Vextor Length:" + vector.size());
        out.println("Vector Shape:" + Arrays.toString(vector.shape().toArray()));

        //Matrices
        NDArray matrixThreeByTwo = NDArray.arange(0, 6, 1, 1, context, DType.Int32());
        matrixThreeByTwo = matrixThreeByTwo.reshape(new int[]{3, 2});
        
        out.println("Matrix:" + matrixThreeByTwo.toString());
        out.println("Matrix:" + matrixThreeByTwo.T().toString());
        //Matrix are  equal to their T // A==A.T if they symmetric 
        NDArray symmetricMatrix = new NDArray(new float[]{1, 2, 3, 2, 0, 4, 3, 4, 5}, new Shape(new int[]{3, 3}), context);
        out.println("A==A.T" + NDArray.equal(symmetricMatrix, symmetricMatrix.T()).toString());

        //Tnsors 
        NDArray tensor = NDArray.arange(0, 24, 1, 1, context, DType.Float64());
        tensor = tensor.reshape(new int[]{2, 3, 4});
        out.println("Matrix:" + tensor.toString());
        
        out.println("===");
        //Basic Properties of Tensor Arithmetic
        NDArray a = NDArray.arange(0, 6, 1, 1, context, DType.Float64());
        a = a.reshape(new int[]{2, 3});
        out.println("Matrix A:" + a.toString());
        NDArray b = null;
        b = a.copy();
        out.println("Matrix B:" + b.toString());
        out.println("X+Y: " + a.add(b).toString());
        out.println("X*Y: " + a.multiply(b).toString());
        out.println("===");
        //Tensor * scalar ;
        out.println("2+Tensor: " + tensor.add(2).toString());
        out.println("2*Tensor: " + tensor.multiply(2).shape().toString());
        out.println("===");

        //Reduction
        sumParam sumParam = new sumParam(vector);
        out.println("Vectors:" + Arrays.toString(vector.toArray()));
        out.println("vertor Sum: " + Arrays.toString(NDArray.sum(sumParam)[0].toArray()));
        out.println("===");
        sumParam = new sumParam(a);
        out.println("Matrix Shape:" + Arrays.toString(a.shape().toArray()));
        out.println("Matrix Sum: " + Arrays.toString(NDArray.sum(sumParam)[0].toArray()));
        sum_axisParam p1 = new sum_axisParam(a);
        out.println("===");
        p1.setAxis(new Shape(List.of(0)));
        //out.println("Shape " + Arrays.toString(p1.getAxis().shape().toArray()));
        out.println("Matrix Sum axis zero Shape: " + Arrays.toString(NDArray.sum_axis(p1)[0].shape().toArray()));
        p1.setAxis(new Shape(List.of(1)));
        out.println("Matrix Sum axis one Shape: " + Arrays.toString(NDArray.sum_axis(p1)[0].shape().toArray()));
        p1.setAxis(new Shape(List.of(0, 1)));
        out.println("Matrix Sum axis one and Zero == array.Sum : " + NDArray.equal(NDArray.sum_axis(p1)[0], a.sum(sumParam)[0]).toString());
        out.println("===");
        meanParam meanParam = new meanParam(a);
        //Matrix Mean == Matrix Sum/Size
        out.println("Matrix Mean" + NDArray.mean(meanParam)[0].toString());
        out.println("Matrixc Sum/size" + NDArray.sum(sumParam)[0].div(a.size()).toString());
        //Matrix Mean == Matrix Sum/Size // Using 0 Axis
        meanParam.setAxis(new Shape(List.of(0)));
        p1.setAxis(new Shape(List.of(0)));
        out.println("===");
        out.println("Matrix Mean axis 0:" + NDArray.mean(meanParam)[0].toString());
        out.println("Matrixc Sum/size axis 0:" + NDArray.sum_axis(p1)[0].div(a.shape().toArray()[0]).toString());
        out.println("===");
        //Non-Reduction Sum 
        p1.setAxis(new Shape(new int[]{1}));
        p1.setKeepdims(true);
        out.println("Sum a with keepdims True allong axis 0: " + NDArray.sum_axis(p1)[0].toString());
        out.println("Sum a / Sum Sof A axis 1: //Also known as boardcast" + NDArray.broadcast_div(a, NDArray.sum_axis(p1)[0], null)[0].toString());
        cumsumParam c = new cumsumParam(a);
        c.setAxis(0);
        out.println("Cumulative sum: " + NDArray.cumsum(c)[0].toString());
        out.println("===");

        //Dot Products
        y = NDArray.ones(new Shape(List.of(3)), context, DType.Float64());
        x = NDArray.arange(0, 3, 1, 1, context, DType.Float64());
        out.println("Y: " + y.toString());
        out.println("X: " + x.toString());
        dotParam dtprm = new dotParam(y, x);
        out.println("y . x : " + NDArray.dot(dtprm)[0].toString());
        sumParam = new sumParam(y.multiply(x));
        out.println("Sum of x * y: " + NDArray.sum(sumParam)[0].toString());
        out.println("===");

        //Matrix–Vector Products
        dtprm = new dotParam(a, x);
        out.println("A Shapre" + Arrays.toString(a.shape().toArray()) + " X Shapre:" + Arrays.toString(x.shape().toArray()));
        out.println("A dot x : " + NDArray.dot(dtprm)[0].toString());
        out.println("===");

        //Norms
        NDArray u = new NDArray(new float[]{3, -4}, new Shape(new int[]{2}), context);
        normParam nrmprm = new normParam(u);
        out.println("Norm of u = {3,-4}" + Arrays.toString(NDArray.norm(nrmprm)[0].toArray()));
        sumParam = new sumParam(u);
        sumParam.setExclude(false);
        out.println("Norm2 of u = {3,-4}" + Arrays.toString(NDArray.sum(sumParam)[0].toArray()));
        NDArray abs_u = NDArray.abs(u, null)[0];
        out.println("abs of u : " + abs_u.toString());
        sumParam = new sumParam(abs_u);
        out.println("Norm1 of u = {3,-4}" + Arrays.toString(NDArray.sum(sumParam)[0].toArray()));
        
        NDArray f = NDArray.ones(context, List.of(4, 9));
        out.println("ones f : " + f.shape().toString());
        nrmprm = new normParam(f);
        out.println("Norm of f" + Arrays.toString(NDArray.norm(nrmprm)[0].toArray()));
        
        exit(0);
        
    }
    
}
